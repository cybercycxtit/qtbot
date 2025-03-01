import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
import requests
import os
from onnx import helper
import onnxruntime as ort
import logging
from utils import fetch_x_sentiment, initialize_mt5
import MetaTrader5 as mt5

# 配置日志
logging.basicConfig(
    filename="logs/train.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 模型路径
TEACHER_MODEL_PATH = "deepseek-v3"  # 假设为预训练模型路径
STUDENT_MODEL_PATH = "models/deepseek_7b"
OPTIMIZED_MODEL_PATH = "models/deepseek_7b_optimized.onnx"

# SiliconFlow API 配置（需替换为实际密钥和端点）
SILICONFLOW_API_KEY = "your_siliconflow_api_key"
SILICONFLOW_API_ENDPOINT = "https://api.siliconflow.com/distill"  # 假设端点

def fetch_historical_data(product, timeframe, bars=1000):
    """从 MT5 获取历史交易数据"""
    if not initialize_mt5():
        logging.error("Failed to initialize MT5")
        return pd.DataFrame()
    
    timeframe_map = {"m1": mt5.TIMEFRAME_M1, "h1": mt5.TIMEFRAME_H1, "d1": mt5.TIMEFRAME_D1}
    mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
    
    rates = mt5.copy_rates_from_pos(product, mt5_timeframe, 0, bars)
    if rates is None:
        logging.error(f"Failed to fetch data for {product}")
        return pd.DataFrame()
    
    return pd.DataFrame(rates)[["time", "open", "high", "low", "close"]]

def generate_synthetic_data(historical_data, num_samples=1000):
    """生成合成数据"""
    mean_returns = historical_data["close"].pct_change().mean()
    std_returns = historical_data["close"].pct_change().std()
    synthetic_returns = np.random.normal(mean_returns, std_returns, num_samples)
    synthetic_prices = [historical_data["close"].iloc[-1]]
    for r in synthetic_returns:
        synthetic_prices.append(synthetic_prices[-1] * (1 + r))
    return pd.DataFrame({"close": synthetic_prices[1:]})

def combine_multimodal_data(historical_data, x_posts):
    """融合历史数据与 X 平台情绪数据"""
    sentiment_score = fetch_x_sentiment(x_posts)
    return pd.DataFrame({
        "close": historical_data["close"],
        "sentiment": [sentiment_score] * len(historical_data["close"])
    })

def distill_with_siliconflow(teacher_model, student_model, data):
    """通过 SiliconFlow API 进行模型蒸馏"""
    # 准备数据（简化为序列化输入）
    input_data = data.to_json()
    
    # 调用 SiliconFlow API（假设格式）
    payload = {
        "teacher_model": teacher_model.config.to_json_string(),
        "student_model": student_model.config.to_json_string(),
        "data": input_data,
        "api_key": SILICONFLOW_API_KEY
    }
    
    try:
        response = requests.post(SILICONFLOW_API_ENDPOINT, json=payload)
        response.raise_for_status()
        distilled_model_weights = response.json()["weights"]
        # 更新学生模型权重（假设返回权重格式兼容）
        student_model.load_state_dict(distilled_model_weights)
        logging.info("Model distillation via SiliconFlow succeeded")
    except Exception as e:
        logging.error(f"SiliconFlow API distillation failed: {e}")
        raise
    
    return student_model

def train_student_locally(teacher_model, student_model, data, epochs=1):
    """本地蒸馏训练（备选方案）"""
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-5)
    criterion = nn.KLDivLoss(reduction="batchmean")
    
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_PATH)
    inputs = tokenizer(data["close"].astype(str).tolist(), return_tensors="pt", padding=True, truncation=True)
    
    for epoch in range(epochs):
        teacher_outputs = teacher_model(**inputs).logits
        student_outputs = student_model(**inputs).logits
        
        loss = criterion(
            nn.functional.log_softmax(student_outputs, dim=-1),
            nn.functional.softmax(teacher_outputs, dim=-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
    
    return student_model

def optimize_model(model, tokenizer, output_path=OPTIMIZED_MODEL_PATH):
    """将模型转换为 ONNX 格式以优化推理速度"""
    dummy_input = tokenizer("dummy input", return_tensors="pt")
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        output_path,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}}
    )
    logging.info(f"Model optimized and saved to {output_path}")

def train_online(model_path, market_data, x_posts, incremental=True):
    """在线学习模式"""
    student_model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    combined_data = combine_multimodal_data(market_data, x_posts)
    inputs = tokenizer(combined_data["close"].astype(str).tolist(), return_tensors="pt", padding=True, truncation=True)
    
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-5)
    outputs = student_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    student_model.save_pretrained("models/deepseek_7b_updated")
    tokenizer.save_pretrained("models/deepseek_7b_updated")
    logging.info("Online learning update completed")

def main():
    """主训练流程"""
    # 1. 加载教师和学生模型
    try:
        teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_PATH)
        student_model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_PATH)
        logging.info("Teacher and student models loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load models: {e}")
        return

    # 2. 获取历史数据并进行数据增强
    historical_data = fetch_historical_data("BTCUSD", "h1")
    if historical_data.empty:
        logging.error("No historical data available")
        return
    
    synthetic_data = generate_synthetic_data(historical_data)
    augmented_data = pd.concat([historical_data, synthetic_data], ignore_index=True)
    x_posts = ["BTC is great!", "Sell BTC now"]  # 示例数据，需替换为真实 X 数据
    training_data = combine_multimodal_data(augmented_data, x_posts)
    logging.info("Data augmentation and multimodal combination completed")

    # 3. 模型蒸馏
    try:
        student_model = distill_with_siliconflow(teacher_model, student_model, training_data)
    except Exception as e:
        logging.warning(f"SiliconFlow distillation failed, falling back to local training: {e}")
        student_model = train_student_locally(teacher_model, student_model, training_data)

    # 4. 保存初始训练模型
    student_model.save_pretrained(STUDENT_MODEL_PATH)
    tokenizer.save_pretrained(STUDENT_MODEL_PATH)
    logging.info(f"Initial distilled model saved to {STUDENT_MODEL_PATH}")

    # 5. 优化推理速度
    optimize_model(student_model, tokenizer)
    
    # 6. 在线学习示例
    latest_data = fetch_historical_data("BTCUSD", "h1", bars=10)  # 获取最新数据
    train_online(STUDENT_MODEL_PATH, latest_data, x_posts)

if __name__ == "__main__":
    main()
