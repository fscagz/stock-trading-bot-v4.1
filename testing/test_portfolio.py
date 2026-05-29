import pandas as pd
from portfolio import analyze_trades

# Sample backtest data
data = {
    "signal": [0, 1, 0, -1, 0, 1, 0],
    "close":  [100, 102, 101, 99, 98, 99, 102],
    "trade_pnl": [0.0, 0.0, -1.0, 0.0, 2.0, 0.0, 2.0],
    "cumulative_pnl": [0.0, 0.0, -1.0, -1.0, 1.0, 1.0, 3.0]
}
index = pd.date_range("2025-06-02 09:30", periods=7, freq="5min")
df = pd.DataFrame(data, index=index)

# Analyze trades with verbose output
metrics = analyze_trades(df, verbose=True)

# Print metrics
print("\n=== Portfolio Metrics ===")
for key, val in metrics.items():
    print(f"{key}: {val}")
