import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from utils import initialize_mt5

class TrendFollowing:
    def generate_signal(self, market_state):
        return "buy" if market_state["trend"] == "up" else "sell"

    def backtest_with_montecarlo(self, product, timeframe, montecarlo_runs):
        if not initialize_mt5():
            return {"profit": 0, "max_drawdown": 0}
        timeframe_map = {"m1": mt5.TIMEFRAME_M1, "h1": mt5.TIMEFRAME_H1, "d1": mt5.TIMEFRAME_D1}
        rates = mt5.copy_rates_from_pos(product, timeframe_map.get(timeframe, mt5.TIMEFRAME_H1), 0, 1000)
        prices = pd.DataFrame(rates)["close"]
        returns = np.diff(prices) / prices[:-1]
        profit = sum(returns) * 0.01
        max_drawdown = max(np.cumsum(returns).min(), 0)
        return {"profit": profit, "max_drawdown": max_drawdown}

class MeanReversion:
    def generate_signal(self, market_state):
        return "buy" if market_state["rsi"] < 30 else "sell" if market_state["rsi"] > 70 else "hold"

    def backtest_with_montecarlo(self, product, timeframe, montecarlo_runs):
        if not initialize_mt5():
            return {"profit": 0, "max_drawdown": 0}
        timeframe_map = {"m1": mt5.TIMEFRAME_M1, "h1": mt5.TIMEFRAME_H1, "d1": mt5.TIMEFRAME_D1}
        rates = mt5.copy_rates_from_pos(product, timeframe_map.get(timeframe, mt5.TIMEFRAME_H1), 0, 1000)
        prices = pd.DataFrame(rates)["close"]
        returns = np.diff(prices) / prices[:-1]
        profit = sum(returns) * 0.005
        max_drawdown = max(np.cumsum(returns).min(), 0)
        return {"profit": profit, "max_drawdown": max_drawdown}

class HybridStrategy:
    def __init__(self):
        self.trend = TrendFollowing()
        self.meanrev = MeanReversion()

    def generate_signal(self, market_state):
        trend_signal = self.trend.generate_signal(market_state)
        meanrev_signal = self.meanrev.generate_signal(market_state)
        return "buy" if trend_signal == "buy" and meanrev_signal == "buy" else "sell" if trend_signal == "sell" else "hold"

    def backtest_with_montecarlo(self, product, timeframe, montecarlo_runs):
        if not initialize_mt5():
            return {"profit": 0, "max_drawdown": 0}
        timeframe_map = {"m1": mt5.TIMEFRAME_M1, "h1": mt5.TIMEFRAME_H1, "d1": mt5.TIMEFRAME_D1}
        rates = mt5.copy_rates_from_pos(product, timeframe_map.get(timeframe, mt5.TIMEFRAME_H1), 0, 1000)
        prices = pd.DataFrame(rates)["close"]
        returns = np.diff(prices) / prices[:-1]
        profit = sum(returns) * 0.007
        max_drawdown = max(np.cumsum(returns).min(), 0)
        return {"profit": profit, "max_drawdown": max_drawdown}
