import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from utils import MarketStateClassifier, initialize_mt5, fetch_x_sentiment
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# AI 模型路径（与 train.py 一致）
MODEL_PATH = "models/deepseek_7b"

class BaseStrategy:
    """策略基类"""
    def __init__(self):
        self.classifier = MarketStateClassifier()
        self.model, self.tokenizer = self.load_ai_model()

    def load_ai_model(self):
        """加载 DeepSeek-7B 模型"""
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            return model, tokenizer
        except Exception as e:
            print(f"Failed to load AI model: {e}")
            return None, None

    def generate_ai_signal(self, data):
        """生成 AI 交易信号"""
        if self.model is None or self.tokenizer is None:
            return "hold"
        inputs = self.tokenizer(data["close"].astype(str).tolist(), return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs).logits
        signal = "buy" if outputs.mean() > 0 else "sell"  # 简化逻辑，需根据实际模型调整
        return signal

    def fetch_data(self, product, timeframe, bars=100):
        """获取 MT5 数据"""
        if not initialize_mt5():
            return pd.DataFrame()
        timeframe_map = {"m1": mt5.TIMEFRAME_M1, "h1": mt5.TIMEFRAME_H1, "d1": mt5.TIMEFRAME_D1}
        rates = mt5.copy_rates_from_pos(product, timeframe_map.get(timeframe, mt5.TIMEFRAME_H1), 0, bars)
        if rates is None:
            return pd.DataFrame()
        return pd.DataFrame(rates)

    def calculate_indicators(self, data):
        """计算技术指标"""
        ema_20 = data["close"].ewm(span=20, adjust=False).mean()
        rsi = self.classifier.calculate_rsi(data["close"])
        
        # MACD
        ema_fast = data["close"].ewm(span=12, adjust=False).mean()
        ema_slow = data["close"].ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=9, adjust=False).mean()
        
        # KDJ
        low_min = data["low"].rolling(window=9).min()
        high_max = data["high"].rolling(window=9).max()
        rsv = (data["close"] - low_min) / (high_max - low_min) * 100
        k = rsv.ewm(span=3, adjust=False).mean()
        d = k.ewm(span=3, adjust=False).mean()
        j = 3 * k - 2 * d
        
        return {
            "ema_20": ema_20.iloc[-1],
            "rsi": rsi,
            "macd": macd.iloc[-1] - signal_line.iloc[-1],
            "kdj_j": j.iloc[-1]
        }

    def backtest_with_montecarlo(self, product, timeframe, montecarlo_runs=100):
        """蒙特卡洛回测"""
        data = self.fetch_data(product, timeframe, bars=1000)
        if data.empty:
            return {"profit": 0, "max_drawdown": 0}

        returns = data["close"].pct_change().dropna()
        mean_ret = returns.mean()
        std_ret = returns.std()

        profits = []
        for _ in range(montecarlo_runs):
            sim_returns = np.random.normal(mean_ret, std_ret, len(returns))
            sim_cum_returns = np.cumprod(1 + sim_returns) - 1
            profit = sim_cum_returns[-1]
            max_drawdown = min(sim_cum_returns)
            profits.append({"profit": profit, "max_drawdown": max_drawdown})
        
        avg_profit = np.mean([p["profit"] for p in profits])
        avg_max_drawdown = np.mean([p["max_drawdown"] for p in profits])
        return {"profit": avg_profit, "max_drawdown": avg_max_drawdown}

class TrendFollowing(BaseStrategy):
    """趋势跟随策略"""
    def generate_signal(self, market_state):
        data = self.fetch_data("BTCUSD", "h1")  # 默认产品和时间框架，可扩展
        if data.empty:
            return "hold"
        
        indicators = self.calculate_indicators(data)
        ai_signal = self.generate_ai_signal(data)
        
        # 趋势跟随：EMA 和 MACD 确认趋势
        if indicators["ema_20"] < data["close"].iloc[-1] and indicators["macd"] > 0:
            return "buy"
        elif indicators["ema_20"] > data["close"].iloc[-1] and indicators["macd"] < 0:
            return "sell"
        return ai_signal if ai_signal != "hold" else "hold"

class MeanReversion(BaseStrategy):
    """均值回归策略"""
    def generate_signal(self, market_state):
        data = self.fetch_data("BTCUSD", "h1")
        if data.empty:
            return "hold"
        
        indicators = self.calculate_indicators(data)
        ai_signal = self.generate_ai_signal(data)
        
        # 均值回归：RSI 和 KDJ 判断超买超卖
        if indicators["rsi"] < 30 and indicators["kdj_j"] < 20:
            return "buy"
        elif indicators["rsi"] > 70 and indicators["kdj_j"] > 80:
            return "sell"
        return ai_signal if ai_signal != "hold" else "hold"

class HybridStrategy(BaseStrategy):
    """混合策略"""
    def __init__(self):
        super().__init__()
        self.trend = TrendFollowing()
        self.meanrev = MeanReversion()

    def generate_signal(self, market_state):
        data = self.fetch_data("BTCUSD", "h1")
        if data.empty:
            return "hold"
        
        trend_signal = self.trend.generate_signal(market_state)
        meanrev_signal = self.meanrev.generate_signal(market_state)
        ai_signal = self.generate_ai_signal(data)
        
        # 混合逻辑：趋势和均值回归一致时优先，AI 信号作为补充
        if trend_signal == meanrev_signal and trend_signal in ["buy", "sell"]:
            return trend_signal
        return ai_signal if ai_signal != "hold" else "hold"

    def backtest_with_montecarlo(self, product, timeframe, montecarlo_runs=100):
        """混合策略回测"""
        trend_results = self.trend.backtest_with_montecarlo(product, timeframe, montecarlo_runs)
        meanrev_results = self.meanrev.backtest_with_montecarlo(product, timeframe, montecarlo_runs)
        
        # 平均结果（可根据实际需求调整权重）
        profit = (trend_results["profit"] + meanrev_results["profit"]) / 2
        max_drawdown = max(trend_results["max_drawdown"], meanrev_results["max_drawdown"])
        return {"profit": profit, "max_drawdown": max_drawdown}

if __name__ == "__main__":
    # 测试策略
    strategy = HybridStrategy()
    state = {"trend": "up", "rsi": 40, "volatility": 0.02}  # 示例状态
    signal = strategy.generate_signal(state)
    print(f"Hybrid Strategy Signal: {signal}")
    
    backtest_results = strategy.backtest_with_montecarlo("BTCUSD", "h1")
    print(f"Backtest Results: Profit={backtest_results['profit']:.2%}, Max Drawdown={backtest_results['max_drawdown']:.2%}")
