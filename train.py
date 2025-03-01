import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import fetch_x_sentiment
import pandas as pd

MODEL_PATH = "models/deepseek_7b"

def train_online(model_path, market_data, x_posts):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sentiment_data = fetch_x_sentiment(x_posts)
    combined_data = combine_data(market_data, sentiment_data)

    inputs = tokenizer(str(combined_data), return_tensors="pt")
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer.step()
    optimizer.zero_grad()

    model.save_pretrained("models/deepseek_7b_updated")
    tokenizer.save_pretrained("models/deepseek_7b_updated")
    print("Model updated successfully.")

def combine_data(market_data, sentiment_data):
    return pd.DataFrame({
        "close": market_data["close"],
        "sentiment": [sentiment_data] * len(market_data["close"])
    })

if __name__ == "__main__":
    market_data = {"close": [100, 101, 102]}
    x_posts = ["BTC is great!", "Sell BTC now"]
    train_online(MODEL_PATH, market_data, x_posts)
