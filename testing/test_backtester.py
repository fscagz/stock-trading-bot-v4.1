# test_backtester.py

from signal_generator import generate_signal
from backtester import backtest_signals
import pandas as pd

# sample data:
data = {
    "close": [100, 102, 101, 98, 97, 99, 100],
    "vwap": [101, 101, 101, 100, 99, 100, 101],
    "sma_20": [99, 99, 99, 99, 99, 99, 99]
}

index = pd.date_range("2025-06-02 09:30", periods=7, freq="5min")
df = pd.DataFrame(data, index = index)

df = generate_signal(df)
df = backtest_signals(df)

print(df[["close", "signal", "entry_price", "exit_price", "trade_pnl", "cumulative_pnl"]])

