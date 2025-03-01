import argparse
import asyncio
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import MetaTrader5 as mt5
from utils import MarketStateClassifier, fetch_x_sentiment, initialize_mt5
from strategy import TrendFollowing, MeanReversion, HybridStrategy
from execution import execute_trade
from risk_management import check_circuit_breaker, adjust_sl_tp
from config import ACCOUNT_ID, API_KEY, SERVER

# 技术指标计算函数
def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = data["close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_kdj(data, n=9, k_period=3, d_period=3):
    low_min = data["low"].rolling(window=n).min()
    high_max = data["high"].rolling(window=n).max()
    rsv = (data["close"] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(span=k_period, adjust=False).mean()
    d = k.ewm(span=d_period, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

def calculate_ema(data, period=5):
    return data["close"].ewm(span=period, adjust=False).mean()

async def fetch_data_async(product, timeframe, bars=50):
    """异步获取 MT5 数据"""
    if not initialize_mt5():
        print("MT5 initialization failed")
        return pd.DataFrame()
    
    timeframe_map = {"m1": mt5.TIMEFRAME_M1, "h1": mt5.TIMEFRAME_H1, "d1": mt5.TIMEFRAME_D1}
    mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
    
    await asyncio.sleep(0.1)  # 模拟异步延迟
    rates = mt5.copy_rates_from_pos(product, mt5_timeframe, 0, bars)
    if rates is None:
        print(f"Failed to fetch data for {product}")
        return pd.DataFrame()
    
    return pd.DataFrame(rates)

def load_ai_model(model_path):
    """加载 DeepSeek-7B 模型"""
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load AI model: {e}")
        return None, None

async def generate_ai_signal(model, tokenizer, data):
    """生成 AI 交易信号"""
    if model is None or tokenizer is None:
        return "hold"
    
    inputs = tokenizer(data["close"].astype(str).tolist(), return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs).logits
    signal = "buy" if outputs.mean() > 0 else "sell"  # 简化逻辑，需根据实际模型调整
    return signal

async def run_cli(args):
    """CLI 主逻辑"""
    # 初始化 MT5
    if not initialize_mt5():
        print("MT5 initialization failed")
        return

    # 获取市场数据
    data = await fetch_data_async(args.product, args.timeframe)
    if data.empty:
        return

    # 计算技术指标
    classifier = MarketStateClassifier()
    rsi = classifier.calculate_rsi(data["close"])
    macd, signal_line = calculate_macd(data)
    k, d, j = calculate_kdj(data)
    ema_5 = calculate_ema(data, period=5)
    
    indicators = {
        1: ("MACD", macd.iloc[-1] - signal_line.iloc[-1]),
        2: ("KDJ", j.iloc[-1]),
        3: ("EMA_5", ema_5.iloc[-1] - data["close"].iloc[-1])
    }
    selected_indicator = indicators.get(args.indicator, ("RSI", rsi))

    # 市场状态
    state = classifier.get_market_state(args.product, args.timeframe)
    print(f"Market State: Trend={state['trend']}, Volatility={state['volatility']:.4f}, RSI={rsi:.2f}")
    print(f"Selected Indicator: {selected_indicator[0]} = {selected_indicator[1]:.4f}")

    # 多策略选择
    if args.strategy == 1:
        strategy = TrendFollowing()
    elif args.strategy == 2:
        strategy = MeanReversion()
    elif args.strategy == 4:
        strategy = HybridStrategy()
    else:
        strategy = HybridStrategy()  # 默认混合策略

    # AI 信号
    model, tokenizer = load_ai_model(args.model)
    ai_signal = await generate_ai_signal(model, tokenizer, data) if args.aipdt else "hold"
    print(f"AI Signal: {ai_signal}")

    # 情绪分析（模拟 FinBERT）
    x_posts = ["BTC is great!", "Sell BTC now"]  # 需替换为真实数据
    sentiment = fetch_x_sentiment(x_posts)
    print(f"Market Sentiment: {sentiment:.2f}")

    # 风控检查
    if not check_circuit_breaker(args.product, args.timeframe):
        print("Trading halted due to circuit breaker")
        return

    # 回测
    if args.backtest:
        results = strategy.backtest_with_montecarlo(args.product, args.timeframe, montecarlo_runs=args.montecarlo)
        print(f"Backtest Results: Profit={results['profit']:.2%}, Max Drawdown={results['max_drawdown']:.2%}")
        return

    # 交易信号
    strategy_signal = strategy.generate_signal(state)
    final_signal = ai_signal if args.aipdt and ai_signal != "hold" else strategy_signal
    print(f"Final Signal: {final_signal}")

    # 执行交易
    if not args.aipdtonly and final_signal in ["buy", "sell"]:
        slippage = adjust_sl_tp(final_signal, 0.01)  # 假设滑点
        execute_trade(final_signal, args.hand, args.product)
    elif args.aipdtonly:
        print("AI prediction only mode: No trade executed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Trading Bot CLI")
    parser.add_argument("--product", "-p", type=str, default="BTCUSD", help="Trading product (e.g., BTCUSD)")
    parser.add_argument("--timeframe", "-f", type=str, default="h1", help="Timeframe (e.g., h1, m15)")
    parser.add_argument("--aipdt", "-ai", type=int, default=1, choices=[0, 1], help="Enable AI prediction (1: yes, 0: no)")
    parser.add_argument("--backtest", "-bt", action="store_true", help="Run backtest")
    parser.add_argument("--aipdtonly", "-aip", action="store_true", help="AI prediction only, no trading")
    parser.add_argument("--indicator", "-idct", type=int, default=1, choices=[1, 2, 3], 
                        help="Indicator: 1=MACD, 2=KDJ, 3=5-day EMA")
    parser.add_argument("--hand", "-hnd", type=float, default=0.01, help="Trade volume (lots)")
    parser.add_argument("--model", "-mdl", type=str, default="models/deepseek_7b", help="AI model path")
    parser.add_argument("--strategy", "-s", type=int, default=4, choices=[1, 2, 4], 
                        help="Strategy: 1=Trend, 2=MeanRev, 4=Hybrid")
    parser.add_argument("--montecarlo", "-mc", type=int, default=100, help="Monte Carlo runs for backtest")
    
    args = parser.parse_args()
    asyncio.run(run_cli(args))
