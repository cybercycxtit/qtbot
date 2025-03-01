from PyQt5 import QtWidgets
from utils import MarketStateClassifier
import sys

class TradingGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.classifier = MarketStateClassifier()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Quant Trading Bot")
        self.setGeometry(100, 100, 600, 400)

        self.state_label = QtWidgets.QLabel("Market State: Unknown", self)
        self.state_label.move(10, 10)

        self.strategy_combo = QtWidgets.QComboBox(self)
        self.strategy_combo.addItems(["Trend", "MeanRev", "Hybrid"])
        self.strategy_combo.move(10, 40)

        self.update_btn = QtWidgets.QPushButton("Update State", self)
        self.update_btn.move(10, 70)
        self.update_btn.clicked.connect(self.update_market_state)

        self.update_market_state()

    def update_market_state(self):
        state = self.classifier.get_market_state("BTCUSD", "h1")
        self.state_label.setText(
            f"Market State: Trend={state['trend']}, Volatility={state['volatility']:.4f}, RSI={state['rsi']:.2f}"
        )

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = TradingGUI()
    gui.show()
    sys.exit(app.exec_())
