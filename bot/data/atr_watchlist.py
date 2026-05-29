"""
ATR-based watchlist for the legacy intraday strategy.

Uses S&P 500 from universe.get_tickers_sp500; computes ATR and returns
top N by volatility. Systematic pipeline uses data.universe.get_universe instead.
"""

import time
import yfinance as yf
import pandas as pd

from data.universe import get_tickers_sp500


def get_sp500_tickers():
    """S&P 500 tickers (Yahoo format). Delegates to universe.get_tickers_sp500."""
    return get_tickers_sp500()


def compute_atr(ticker, period=14):
    """
    Calculates the 14-day ATR for a ticker using historical daily data.
    Returns float or None if insufficient data.
    """
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if df.empty or len(df) < period + 1:
            return None
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(0, axis=1)
        # yfinance returns Open, High, Low, Close
        high = df["High"] if "High" in df.columns else df["high"]
        low = df["Low"] if "Low" in df.columns else df["low"]
        close = df["Close"] if "Close" in df.columns else df["close"]
        df["H-L"] = high - low
        df["H-PC"] = abs(high - close.shift(1))
        df["L-PC"] = abs(low - close.shift(1))
        df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
        atr = df["TR"].rolling(window=period).mean().iloc[-1]
        return float(atr)
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


def get_top_atr_stocks(top_n, batch_size=50):
    """
    Returns top N S&P 500 stocks by ATR (highest volatility).
    Input: top_n (int), batch_size (int, default 50)
    Output: List of (ticker, ATR) tuples sorted by ATR descending.
    """
    tickers = get_tickers_sp500()
    all_results = []

    print(f"Scanning {len(tickers)} tickers in batches of {batch_size}...")

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"Processing batch {i // batch_size + 1}...")
        for ticker in batch:
            atr = compute_atr(ticker)
            if atr is not None:
                all_results.append((ticker, atr))
        time.sleep(1)

    sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)
    print(f"\nTop {top_n} S&P 500 Stocks by ATR:")
    for ticker, atr in sorted_results[:top_n]:
        print(f"{ticker}: ATR = {atr:.2f}")

    return sorted_results[:top_n]


if __name__ == "__main__":
    get_top_atr_stocks(25)
