import yfinance as yf

# Download 15-minute interval data for SPY for the past 5 days
df = yf.download("SPY", interval="15m", period="5d")

print(df.head())
