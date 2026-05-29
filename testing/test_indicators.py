from data.data_loader_yf import get_5min_data
from indicators import calculate_vwap, calculate_intraday_sma
import matplotlib.pyplot as plt

# Step 1: Load 5-minute data for SPY
df = get_5min_data("AAPL", days_back=3)

print("Columns in DataFrame:", df.columns)
print(df.head())

# Step 2: Calculate indicators
print("Calculating VWAP...")
df["vwap"] = calculate_vwap(df)
print("VWAP calculated. Columns now:", df.columns)
print(df[["close", "vwap"]].head())

df["SMA_20"] = calculate_intraday_sma(df, window=20)
print("SMA calculated. Columns now:", df.columns)
print(df[["close", "SMA_20"]].dropna().head())


# Step 3: Drop NaNs for clean plotting
df = df.dropna(subset=["vwap", "SMA_20"])

# Step 4: Plot
plt.figure(figsize=(14, 6))

plt.plot(df.index, df["close"], label="Close Price", linewidth=1)
plt.plot(df.index, df["vwap"], label="VWAP", linewidth=1.2)
plt.plot(df.index, df["SMA_20"], label="20-bar SMA", linewidth=1.2)
plt.title("SPY 5-min Close Price, VWAP, and 20-bar SMA")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
