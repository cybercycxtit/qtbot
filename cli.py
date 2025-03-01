import argparse
import pandas as pd
from utils import MarketStateClassifier
from strategy import TrendFollowing, MeanReversion, HybridStrategy
from execution import execute_trade

def run_cli(args):
    classifier = MarketStateClassifier()
    state = classifier.get_market_state(product=args.product, timeframe=args.timeframe)

    if args.strategy == 1:
        strategy = TrendFollowing()
    elif args.strategy == 2:
        strategy = MeanReversion()
    elif args.strategy == 4:
        strategy = HybridStrategy()
    else:
        strategy = HybridStrategy()  # 默认混合策略

    if args.backtest:
        results = strategy.backtest_with_montecarlo(args.product, args.timeframe, montecarlo_runs=args.montecarlo)
        print(f"Backtest Results: Profit={results['profit']:.2%}, Max Drawdown={results['max_drawdown']:.2%}")
    else:
        signal = strategy.generate_signal(state)
        execute_trade(signal, args.hand, args.product)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Trading Bot CLI")
    parser.add_argument("--product", "-p", type=str, default="BTCUSD", help="Trading product (e.g., BTCUSD)")
    parser.add_argument("--timeframe", "-f", type=str, default="h1", help="Timeframe (e.g., h1, m15)")
    parser.add_argument("--strategy", "-s", type=int, default=3, help="1: Trend, 2: MeanRev, 3: AIOnly, 4: Hybrid")
    parser.add_argument("--montecarlo", "-mc", type=int, default=100, help="Monte Carlo runs for backtest")
    parser.add_argument("--hand", "-hnd", type=float, default=0.01, help="Trade volume (lots)")
    args = parser.parse_args()
    run_cli(args)
