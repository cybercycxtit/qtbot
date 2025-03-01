# MT5 配置
MT5_API_KEY = "your_mt5_password"  # MT5 账户密码
MT5_ACCOUNT_ID = 123456789         # MT5 账户 ID
MT5_SERVER = "your_broker_server"  # MT5 经纪商服务器地址，例如 "MetaQuotes-Demo"

# Binance 配置（可选，需替换为真实密钥）
BINANCE_API_KEY = "your_binance_api_key"
BINANCE_SECRET_KEY = "your_binance_secret_key"

# Interactive Brokers (IB) 配置（可选，需替换为真实参数）
IB_HOST = "127.0.0.1"              # IB Gateway 或 TWS 的主机地址
IB_PORT = 7497                     # IB 端口号（TWS: 7496, Gateway: 7497）
IB_CLIENT_ID = 1                   # IB 客户端 ID

# 交易参数
SYMBOLS = ["BTCUSD", "ETHUSD", "EURUSD"]  # 支持的交易产品
TIMEFRAMES = ["m1", "m5", "h1", "d1"]     # 支持的时间框架
DEFAULT_PRODUCT = "BTCUSD"                # 默认交易产品
DEFAULT_TIMEFRAME = "h1"                  # 默认时间框架
DEFAULT_VOLUME = 0.01                     # 默认交易手数

# 风控参数
CIRCUIT_BREAKER_THRESHOLD = 0.05  # 熔断阈值（5% 价格波动）
MAX_DRAWDOWN = 0.1               # 最大回撤限制（10%）
RISK_PER_TRADE = 0.01            # 单笔交易风险比例（1%）

# 策略参数
STRATEGY_OPTIONS = {
    1: "TrendFollowing",
    2: "MeanReversion",
    4: "HybridStrategy"
}
DEFAULT_STRATEGY = 4  # 默认混合策略

# AI 模型参数
AI_MODEL_PATH = "models/deepseek_7b"  # DeepSeek-7B 模型路径

# 技术指标参数
INDICATOR_OPTIONS = {
    1: "MACD",
    2: "KDJ",
    3: "EMA_5"
}
DEFAULT_INDICATOR = 1  # 默认 MACD

# 执行策略参数
EXECUTION_STRATEGIES = ["market", "vwap", "twap"]  # 支持的执行策略
DEFAULT_EXECUTION_STRATEGY = "market"              # 默认市场价执行

# 数据库配置（可选，待启用）
REDIS_HOST = "localhost"
REDIS_PORT = 6379
MONGODB_URI = "mongodb://localhost:27017"
DATABASE_NAME = "quant_trading_db"

# 日志配置
LOG_DIRECTORY = "logs"
LOG_FILE = "trade.log"

# 函数：获取配置
def get_mt5_config():
    """返回 MT5 配置"""
    return {
        "api_key": MT5_API_KEY,
        "account_id": MT5_ACCOUNT_ID,
        "server": MT5_SERVER
    }

def get_binance_config():
    """返回 Binance 配置"""
    return {
        "api_key": BINANCE_API_KEY,
        "secret_key": BINANCE_SECRET_KEY
    }

def get_ib_config():
    """返回 IB 配置"""
    return {
        "host": IB_HOST,
        "port": IB_PORT,
        "client_id": IB_CLIENT_ID
    }

if __name__ == "__main__":
    # 测试配置加载
    print("MT5 Config:", get_mt5_config())
    print("Supported Symbols:", SYMBOLS)
    print("Default Strategy:", STRATEGY_OPTIONS[DEFAULT_STRATEGY])
