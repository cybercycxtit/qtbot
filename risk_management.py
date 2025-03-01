import MetaTrader5 as mt5
import pandas as pd
from utils import initialize_mt5

def check_circuit_breaker(product, timeframe):
    if not initialize_mt5():
        return True
    timeframe_map = {"m1": mt5.TIMEFRAME_M1, "h1": mt5.TIMEFRAME_H1, "d1": mt5.TIMEFRAME_D1}
    rates = mt5.copy_rates_from_pos(product, timeframe_map.get(timeframe, mt5.TIMEFRAME_H1), 0, 6)
    price_history = pd.DataFrame(rates)["close"]
    change = (price_history.iloc[-1] - price_history.iloc[-6]) / price_history.iloc[-6]
    if abs(change) > 0.05:
        print("Circuit breaker triggered!")
        return False
    return True

def adjust_sl_tp(signal, slippage):
    base_sl = 0.02
    adjusted_sl = base_sl * (1 + slippage)
    return adjusted_sl

if __name__ == "__main__":
    print(check_circuit_breaker("BTCUSD", "h1"))
