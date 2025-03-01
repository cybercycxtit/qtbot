import MetaTrader5 as mt5
from utils import predict_slippage, initialize_mt5
from risk_management import adjust_sl_tp

def execute_trade(signal, volume, product):
    if not initialize_mt5():
        print("MT5 not initialized")
        return

    slippage = predict_slippage(product, volume)
    print(f"Executing {signal} for {volume} lots of {product} with slippage {slippage:.4f}")

    symbol_info = mt5.symbol_info(product)
    if symbol_info is None:
        print(f"Symbol {product} not found")
        return

    price = mt5.symbol_info_tick(product).ask if signal == "buy" else mt5.symbol_info_tick(product).bid
    sl = adjust_sl_tp(signal, slippage)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": product,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if signal == "buy" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": price - sl * price if signal == "buy" else price + sl * price,
        "tp": 0.0,
        "deviation": 10,
        "magic": 123456,
        "comment": "Quant Bot Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    if slippage > 0.01:
        split_orders = [volume / 3] * 3
        for order_vol in split_orders:
            request["volume"] = order_vol
            result = mt5.order_send(request)
            print(f"Split order result: {result}")
    else:
        result = mt5.order_send(request)
        print(f"Single order result: {result}")

if __name__ == "__main__":
    execute_trade("buy", 0.03, "BTCUSD")
