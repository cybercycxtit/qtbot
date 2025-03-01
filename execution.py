import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import asyncio
from utils import initialize_mt5, fetch_lob_data, fetch_data_async
from risk_management import RiskManager
import time

class ExecutionEngine:
    """交易执行引擎"""
    def __init__(self):
        self.risk_mgr = RiskManager()

    async def fetch_market_data(self, product, timeframe, bars=100):
        """异步获取市场数据"""
        return await fetch_data_async(product, timeframe, bars)

    def predict_slippage(self, product, volume):
        """基于 LOB 数据预测滑点"""
        lob = fetch_lob_data(product)
        if not lob["bid_volume"] or not lob["ask_volume"]:
            return 0.01  # 默认滑点
        
        bid_depth = sum(lob["bid_volume"])  # 买单深度
        ask_depth = sum(lob["ask_volume"])  # 卖单深度
        total_depth = bid_depth + ask_depth
        
        if total_depth == 0:
            return 0.01
        
        slippage = volume / total_depth  # 简单线性滑点模型
        return min(slippage, 0.05)  # 限制最大滑点 5%

    async def calculate_vwap(self, product, timeframe, duration=60):
        """计算 VWAP"""
        data = await self.fetch_market_data(product, timeframe, bars=duration)
        if data.empty:
            return None
        
        volume = data["tick_volume"] if "tick_volume" in data else pd.Series([1] * len(data))
        vwap = (data["close"] * volume).sum() / volume.sum()
        return vwap

    async def calculate_twap(self, product, timeframe, duration=60):
        """计算 TWAP"""
        data = await self.fetch_market_data(product, timeframe, bars=duration)
        if data.empty:
            return None
        
        twap = data["close"].mean()
        return twap

    def adaptive_execution(self, signal, volume, product, target_participation=0.1):
        """自适应执行策略（POV）"""
        if not initialize_mt5():
            return None
        
        tick = mt5.symbol_info_tick(product)
        current_price = tick.ask if signal == "buy" else tick.bid
        market_volume = tick.volume  # 当前市场成交量
        
        if market_volume == 0:
            return volume
        
        # 根据目标参与率调整订单量
        adjusted_volume = min(volume, market_volume * target_participation)
        return max(adjusted_volume, 0.01)  # 最小手数

    async def execute_trade(self, signal, volume, product, timeframe="h1", strategy="market"):
        """执行交易"""
        if not initialize_mt5():
            print("MT5 not initialized")
            return None
        
        # 风控检查
        slippage = self.predict_slippage(product, volume)
        adjusted_volume, sl, tp = self.risk_mgr.manage_risk(signal, product, timeframe, slippage, volume)
        if adjusted_volume is None:
            print("Trade halted by risk management")
            return None

        symbol_info = mt5.symbol_info(product)
        if symbol_info is None:
            print(f"Symbol {product} not found")
            return None

        # 根据策略选择执行价格
        if strategy == "vwap":
            price = await self.calculate_vwap(product, timeframe)
        elif strategy == "twap":
            price = await self.calculate_twap(product, timeframe)
        else:  # 默认市场价
            tick = mt5.symbol_info_tick(product)
            price = tick.ask if signal == "buy" else tick.bid
        
        if price is None:
            print("Failed to calculate execution price")
            return None

        # 智能分单
        if slippage > 0.01:  # 滑点大于 1%
            split_orders = self.split_order(adjusted_volume, slippage)
            results = []
            for order_vol in split_orders:
                result = self.place_order(signal, order_vol, product, price, sl, tp)
                results.append(result)
                await asyncio.sleep(0.1)  # 降低延迟，模拟分批执行
            return results
        else:
            return self.place_order(signal, adjusted_volume, product, price, sl, tp)

    def split_order(self, volume, slippage):
        """智能分单"""
        num_splits = int(slippage * 100)  # 滑点越大，分单越多
        num_splits = max(2, min(num_splits, 5))  # 限制 2-5 单
        return [volume / num_splits] * num_splits

    def place_order(self, signal, volume, product, price, sl, tp):
        """下单"""
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": product,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if signal == "buy" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": price - sl * price if signal == "buy" else price + sl * price,
            "tp": price + tp * price if signal == "buy" and tp > 0 else price - tp * price if signal == "sell" and tp > 0 else 0,
            "deviation": 10,
            "magic": 123456,
            "comment": f"Quant Bot {strategy}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        start_time = time.time()
        result = mt5.order_send(request)
        latency = time.time() - start_time
        print(f"Order executed: {signal}, Volume: {volume}, Latency: {latency:.4f}s, Result: {result}")
        return result

async def execute_trade_wrapper(signal, volume, product, timeframe="h1", strategy="market"):
    """外部调用接口"""
    engine = ExecutionEngine()
    return await engine.execute_trade(signal, volume, product, timeframe, strategy)

if __name__ == "__main__":
    # 测试执行
    async def test_execution():
        result = await execute_trade_wrapper("buy", 0.03, "BTCUSD", strategy="vwap")
        print(f"VWAP Execution Result: {result}")
        
        result = await execute_trade_wrapper("sell", 0.05, "BTCUSD", strategy="twap")
        print(f"TWAP Execution Result: {result}")
        
        result = await execute_trade_wrapper("buy", 0.1, "BTCUSD", strategy="market")
        print(f"Market Execution Result: {result}")

    asyncio.run(test_execution())
