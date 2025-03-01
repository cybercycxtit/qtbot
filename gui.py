import sys
import asyncio
import aiohttp
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QDockWidget
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from utils import MarketStateClassifier, fetch_x_sentiment, initialize_mt5
from execution import execute_trade
from config import ACCOUNT_ID, API_KEY, SERVER

class TradingGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.classifier = MarketStateClassifier()
        self.loop = asyncio.get_event_loop()
        self.init_ui()
        self.update_task = None

    def init_ui(self):
        self.setWindowTitle("Quant Trading Bot")
        self.setGeometry(100, 100, 1200, 800)

        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 1. 市场数据展示（K 线 + 技术指标）
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        # 2. 可折叠工具栏（Dock Widget）
        self.toolbar_dock = QDockWidget("Tools", self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.toolbar_dock)
        toolbar_widget = QWidget()
        toolbar_layout = QVBoxLayout(toolbar_widget)

        # 输入框：交易产品名称
        self.product_input = QtWidgets.QLineEdit("BTCUSD", self)
        toolbar_layout.addWidget(QtWidgets.QLabel("Product:"))
        toolbar_layout.addWidget(self.product_input)

        # 交易控制：手数、止损、止盈
        self.volume_input = QtWidgets.QDoubleSpinBox(self)
        self.volume_input.setValue(0.01)
        self.sl_input = QtWidgets.QDoubleSpinBox(self)
        self.sl_input.setValue(0.02)
        self.tp_input = QtWidgets.QDoubleSpinBox(self)
        self.tp_input.setValue(0.0)
        toolbar_layout.addWidget(QtWidgets.QLabel("Volume (lots):"))
        toolbar_layout.addWidget(self.volume_input)
        toolbar_layout.addWidget(QtWidgets.QLabel("Stop Loss (%):"))
        toolbar_layout.addWidget(self.sl_input)
        toolbar_layout.addWidget(QtWidgets.QLabel("Take Profit (%):"))
        toolbar_layout.addWidget(self.tp_input)

        # 交易执行按钮
        self.buy_btn = QtWidgets.QPushButton("Buy", self)
        self.sell_btn = QtWidgets.QPushButton("Sell", self)
        self.buy_btn.clicked.connect(self.manual_buy)
        self.sell_btn.clicked.connect(self.manual_sell)
        toolbar_layout.addWidget(self.buy_btn)
        toolbar_layout.addWidget(self.sell_btn)

        # 自动交易开关
        self.auto_trade_checkbox = QtWidgets.QCheckBox("Auto Trading", self)
        toolbar_layout.addWidget(self.auto_trade_checkbox)

        self.toolbar_dock.setWidget(toolbar_widget)

        # 3. 状态显示区域
        self.state_label = QtWidgets.QLabel("Market State: Unknown", self)
        self.signal_label = QtWidgets.QLabel("AI Signal: None", self)
        self.sentiment_label = QtWidgets.QLabel("Market Sentiment: Unknown", self)
        main_layout.addWidget(self.state_label)
        main_layout.addWidget(self.signal_label)
        main_layout.addWidget(self.sentiment_label)

        # 启动异步更新
        self.start_async_update()

    def plot_kline(self, data):
        """绘制 K 线图和技术指标"""
        self.ax.clear()
        closes = data["close"]
        highs = data["high"]
        lows = data["low"]
        opens = data["open"]

        # 绘制 K 线（简化版）
        for i in range(len(data)):
            color = "green" if closes[i] >= opens[i] else "red"
            self.ax.plot([i, i], [lows[i], highs[i]], color="black")  # 影线
            self.ax.plot([i, i], [opens[i], closes[i]], color=color, linewidth=5)  # 实体

        # 计算并绘制 EMA
        ema = data["close"].ewm(span=20, adjust=False).mean()
        self.ax.plot(ema, label="EMA 20", color="blue")
        
        # RSI（副图）
        self.ax2 = self.ax.twinx()
        rsi = self.classifier.calculate_rsi(data["close"])
        self.ax2.plot([len(data) - 1], [rsi], "ro", label=f"RSI: {rsi:.2f}")
        self.ax2.set_ylim(0, 100)
        
        self.ax.legend()
        self.canvas.draw()

    async def update_data(self):
        """异步更新市场数据"""
        while True:
            if not initialize_mt5():
                self.state_label.setText("Market State: MT5 Connection Failed")
                await asyncio.sleep(5)
                continue

            product = self.product_input.text()
            timeframe = "h1"  # 可扩展为用户选择
            
            # 获取市场数据
            rates = mt5.copy_rates_from_pos(product, mt5.TIMEFRAME_H1, 0, 50)
            if rates is None:
                self.state_label.setText(f"Market State: Failed to fetch {product}")
                await asyncio.sleep(5)
                continue
            
            data = pd.DataFrame(rates)
            state = self.classifier.get_market_state(product, timeframe)
            
            # 更新状态显示
            self.state_label.setText(
                f"Market State: Trend={state['trend']}, Volatility={state['volatility']:.4f}, RSI={state['rsi']:.2f}"
            )
            
            # 更新 K 线图
            self.plot_kline(data)

            # AI 交易信号（假设从模型获取）
            signal = "buy" if state["trend"] == "up" and state["rsi"] < 70 else "sell" if state["rsi"] > 30 else "hold"
            self.signal_label.setText(f"AI Signal: {signal}")
            if self.auto_trade_checkbox.isChecked():
                self.execute_auto_trade(signal, product)

            # 市场情绪分析
            async with aiohttp.ClientSession() as session:
                # 模拟 X 数据，实际需替换为真实 API 调用
                x_posts = ["BTC is great!", "Sell BTC now"]
                sentiment = fetch_x_sentiment(x_posts)
                self.sentiment_label.setText(f"Market Sentiment: {sentiment:.2f}")

            await asyncio.sleep(5)  # 每 5 秒更新一次

    def start_async_update(self):
        """启动异步更新任务"""
        if self.update_task is None or self.update_task.done():
            self.update_task = self.loop.create_task(self.update_data())

    def manual_buy(self):
        """手动买入"""
        product = self.product_input.text()
        volume = self.volume_input.value()
        execute_trade("buy", volume, product)

    def manual_sell(self):
        """手动卖出"""
        product = self.product_input.text()
        volume = self.volume_input.value()
        execute_trade("sell", volume, product)

    def execute_auto_trade(self, signal, product):
        """自动交易"""
        if signal in ["buy", "sell"]:
            volume = self.volume_input.value()
            execute_trade(signal, volume, product)

    def closeEvent(self, event):
        """关闭窗口时取消异步任务"""
        if self.update_task:
            self.update_task.cancel()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    loop = asyncio.get_event_loop()
    gui = TradingGUI()
    gui.show()
    loop.run_forever()
