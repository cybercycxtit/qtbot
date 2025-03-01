import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio
from config import ACCOUNT_ID, API_KEY, SERVER

# AI 模型路径（与 train.py 一致）
MODEL_PATH = "models/deepseek_7b"

def initialize_mt5():
    """初始化 MT5 连接"""
    if not mt5.initialize(server=SERVER):
        print("MT5 initialization failed")
        return False
    if not mt5.login(ACCOUNT_ID, password=API_KEY):
        print("MT5 login failed")
        return False
    return True

async def fetch_data_async(product, timeframe, bars=100):
    """异步获取 MT5 市场数据"""
    if not initialize_mt5():
        return pd.DataFrame()
    
    timeframe_map = {"m1": mt5.TIMEFRAME_M1, "h1": mt5.TIMEFRAME_H1, "d1": mt5.TIMEFRAME_D1}
    mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
    
    await asyncio.sleep(0.1)  # 模拟异步延迟
    rates = mt5.copy_rates_from_pos(product, mt5_timeframe, 0, bars)
    if rates is None:
        print(f"Failed to fetch data for {product}")
        return pd.DataFrame()
    
    return pd.DataFrame(rates)

def fetch_lob_data(product):
    """获取订单簿数据（简化版）"""
    if not initialize_mt5():
        return {"bid_volume": [0], "ask_volume": [0]}
    
    tick = mt5.symbol_info_tick(product)
    if tick is None:
        return {"bid_volume": [0], "ask_volume": [0]}
    
    # 假设获取最近 5 档深度（MT5 未直接提供完整 LOB，需经纪商支持）
    return {
        "bid_volume": [tick.bid * 10],  # 模拟数据
        "ask_volume": [tick.ask * 10]
    }

def calculate_macd(data, fast=12, slow=26, signal=9):
    """计算 MACD"""
    ema_fast = data["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = data["close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_ema(data, period=20):
    """计算 EMA"""
    return data["close"].ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    """计算 RSI"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def calculate_kdj(data, n=9, k_period=3, d_period=3):
    """计算 KDJ"""
    low_min = data["low"].rolling(window=n).min()
    high_max = data["high"].rolling(window=n).max()
    rsv = (data["close"] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(span=k_period, adjust=False).mean()
    d = k.ewm(span=d_period, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

async def fetch_x_sentiment(posts):
    """分析 X 平台情绪（模拟版）"""
    await asyncio.sleep(0.1)  # 模拟异步延迟
    sentiment_scores = [1 if "great" in p.lower() else -1 for p in posts]
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

def load_ai_model():
    """加载 DeepSeek-7B 模型"""
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load AI model: {e}")
        return None, None

async def generate_ai_prediction(model, tokenizer, data):
    """生成 AI 预测"""
    if model is None or tokenizer is None:
        return "hold"
    
    inputs = tokenizer(data["close"].astype(str).tolist(), return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs).logits
    signal = "buy" if outputs.mean() > 0 else "sell"  # 简化逻辑，需根据实际模型调整
    return signal

class MarketStateClassifier:
    """市场状态分类器"""
    def get_market_state(self, product, timeframe):
        """获取市场状态"""
        data = asyncio.run(fetch_data_async(product, timeframe, bars=20))
        if data.empty:
            return {"volatility": 0, "trend": "unknown", "rsi": 0}

        volatility = data["close"].pct_change().std()
        ema_20 = calculate_ema(data)
        trend = "up" if data["close"].iloc[-1] > ema_20.iloc[-1] else "down"
        rsi = calculate_rsi(data["close"])

        return {"volatility": volatility, "trend": trend, "rsi": rsi}

    def calculate_rsi(self, data, period=14):
        """计算 RSI（复用函数）"""
        return calculate_rsi(data, period)

if __name__ == "__main__":
    # 测试功能
    async def test_utils():
        # 测试 MT5 数据获取
        data = await fetch_data_async("BTCUSD", "h1")
        print("Market Data Sample:")
        print(data.tail())

        # 测试技术指标
        macd, signal_line = calculate_macd(data)
        print(f"MACD: {macd.iloc[-1]:.4f}, Signal: {signal_line.iloc[-1]:.4f}")
        ema = calculate_ema(data)
        print(f"EMA 20: {ema.iloc[-1]:.4f}")
        rsi = calculate_rsi(data["close"])
        print(f"RSI: {rsi:.2f}")
        k, d, j = calculate_kdj(data)
        print(f"KDJ: K={k.iloc[-1]:.2f}, D={d.iloc[-1]:.2f}, J={j.iloc[-1]:.2f}")

        # 测试 LOB 数据
        lob = fetch_lob_data("BTCUSD")
        print(f"LOB Data: {lob}")

        # 测试 X 情绪分析
        posts = ["BTC is great!", "Sell BTC now"]
        sentiment = await fetch_x_sentiment(posts)
        print(f"X Sentiment: {sentiment:.2f}")

        # 测试 AI 预测
        model, tokenizer = load_ai_model()
        signal = await generate_ai_prediction(model, tokenizer, data)
        print(f"AI Prediction: {signal}")

        # 测试市场状态分类器
        classifier = MarketStateClassifier()
        state = classifier.get_market_state("BTCUSD", "h1")
        print(f"Market State: {state}")

    asyncio.run(test_utils())
