import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from config import ACCOUNT_ID, API_KEY, SERVER

def initialize_mt5():
    if not mt5.initialize(server=SERVER):
        print("MT5 initialization failed")
        return False
    if not mt5.login(ACCOUNT_ID, password=API_KEY):
        print("MT5 login failed")
        return False
    return True

class MarketStateClassifier:
    def get_market_state(self, product, timeframe):
        if not initialize_mt5():
            return {"volatility": 0, "trend": "unknown", "rsi": 0}

        timeframe_map = {"m1": mt5.TIMEFRAME_M1, "h1": mt5.TIMEFRAME_H1, "d1": mt5.TIMEFRAME_D1}
        mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)

        rates = mt5.copy_rates_from_pos(product, mt5_timeframe, 0, 20)
        if rates is None:
            print(f"Failed to get data for {product}")
            return {"volatility": 0, "trend": "unknown", "rsi": 0}

        data = pd.DataFrame(rates)
        data["close"] = data["close"]
        data["ema_20"] = data["close"].ewm(span=20, adjust=False).mean()

        volatility = data["close"].pct_change().std()
        trend = "up" if data["close"].iloc[-1] > data["ema_20"].iloc[-1] else "down"
        rsi = self.calculate_rsi(data["close"])

        return {"volatility": volatility, "trend": trend, "rsi": rsi}

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).iloc[-1]

def fetch_lob_data(product):
    if not initialize_mt5():
        return {"bid_volume": [0]}
    tick = mt5.symbol_info_tick(product)
    return {"bid_volume": [tick.bid * 10]}

def fetch_x_sentiment(posts):
    sentiment_scores = [1 if "great" in p.lower() else -1 for p in posts]
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

if __name__ == "__main__":
    classifier = MarketStateClassifier()
    state = classifier.get_market_state("BTCUSD", "h1")
    print(state)
