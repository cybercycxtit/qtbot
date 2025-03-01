import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from utils import initialize_mt5, fetch_data_async
from config import CIRCUIT_BREAKER_THRESHOLD

class RiskManager:
    """风控管理类"""
    def __init__(self, max_drawdown=0.1, default_volume=0.01):
        self.max_drawdown = max_drawdown  # 最大回撤限制（10%）
        self.default_volume = default_volume  # 默认手数
        self.account_balance = self.get_account_balance()

    def get_account_balance(self):
        """获取账户余额"""
        if not initialize_mt5():
            return 0.0
        return mt5.account_balance() or 0.0

    def get_account_equity(self):
        """获取账户净值"""
        if not initialize_mt5():
            return 0.0
        return mt5.account_equity() or 0.0

    async def fetch_historical_data(self, product, timeframe, bars=100):
        """异步获取历史数据"""
        return await fetch_data_async(product, timeframe, bars)

    def check_circuit_breaker(self, product, timeframe):
        """熔断机制：检测极端波动"""
        data = asyncio.run(self.fetch_historical_data(product, timeframe, bars=6))
        if data.empty:
            return True  # 默认不熔断
        
        price_history = data["close"]
        change = (price_history.iloc[-1] - price_history.iloc[-6]) / price_history.iloc[-6]
        if abs(change) > CIRCUIT_BREAKER_THRESHOLD:
            print(f"Circuit breaker triggered: {change:.2%} exceeds {CIRCUIT_BREAKER_THRESHOLD:.2%}")
            return False
        return True

    def calculate_max_drawdown(self):
        """计算当前账户最大回撤"""
        balance = self.get_account_balance()
        equity = self.get_account_equity()
        if balance == 0:
            return 0.0
        drawdown = (balance - equity) / balance
        return drawdown

    def check_drawdown_limit(self):
        """检查是否超过最大回撤限制"""
        drawdown = self.calculate_max_drawdown()
        if drawdown > self.max_drawdown:
            print(f"Max drawdown exceeded: {drawdown:.2%} > {self.max_drawdown:.2%}")
            return False
        return True

    def adjust_sl_tp(self, signal, slippage, product, timeframe):
        """自适应止损/止盈"""
        data = asyncio.run(self.fetch_historical_data(product, timeframe))
        if data.empty:
            return 0.02, 0.0  # 默认止损 2%，无止盈
        
        volatility = data["close"].pct_change().std()
        base_sl = 0.02  # 默认止损 2%
        
        # 根据波动率和滑点动态调整止损
        adjusted_sl = base_sl * (1 + slippage + volatility * 10)  # 波动率放大因子
        adjusted_tp = adjusted_sl * 1.5 if signal == "buy" else 0.0  # 简单止盈策略
        
        return min(adjusted_sl, 0.1), adjusted_tp  # 止损上限 10%

    def calculate_position_size(self, product, timeframe, risk_per_trade=0.01):
        """动态仓位管理"""
        data = asyncio.run(self.fetch_historical_data(product, timeframe))
        if data.empty:
            return self.default_volume
        
        volatility = data["close"].pct_change().std()
        current_price = data["close"].iloc[-1]
        balance = self.get_account_balance()
        
        if balance == 0 or current_price == 0:
            return self.default_volume
        
        # 风险金额 = 账户余额 * 单笔风险比例
        risk_amount = balance * risk_per_trade
        # 每点价值（假设 1 手 = 100,000 单位，需根据产品调整）
        pip_value = 100000 / current_price
        
        # 根据波动率调整仓位
        position_size = risk_amount / (pip_value * volatility * 100)
        adjusted_size = max(min(position_size, self.default_volume * 5), self.default_volume / 10)  # 限制范围
        
        return round(adjusted_size, 2)

    def manage_risk(self, signal, product, timeframe, slippage=0.01, volume=None):
        """综合风控管理"""
        # 检查熔断
        if not self.check_circuit_breaker(product, timeframe):
            return None, None, None  # 暂停交易
        
        # 检查最大回撤
        if not self.check_drawdown_limit():
            return None, None, None  # 暂停交易
        
        # 计算动态仓位
        volume = volume if volume else self.calculate_position_size(product, timeframe)
        
        # 计算自适应止损/止盈
        sl, tp = self.adjust_sl_tp(signal, slippage, product, timeframe)
        
        return volume, sl, tp

if __name__ == "__main__":
    # 测试风控管理
    risk_mgr = RiskManager(max_drawdown=0.1, default_volume=0.01)
    
    # 测试熔断
    print("Circuit Breaker:", risk_mgr.check_circuit_breaker("BTCUSD", "h1"))
    
    # 测试最大回撤
    print("Drawdown Check:", risk_mgr.check_drawdown_limit())
    
    # 测试仓位和止损/止盈
    volume, sl, tp = risk_mgr.manage_risk("buy", "BTCUSD", "h1", slippage=0.01)
    print(f"Position Size: {volume}, SL: {sl:.4f}, TP: {tp:.4f}")
