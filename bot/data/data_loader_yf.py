"""
Intraday OHLCV via Yahoo Finance (5m, 15m, 1h, etc.).

Used by the legacy VWAP strategy. Daily data for the systematic pipeline
is in daily_loader.py.
"""

import yfinance as yf
import pandas as pd


def get_intraday_data(symbol: str, interval: str = "15m", period: str = "5d") -> pd.DataFrame:
    """
    Fetches intraday OHLCV for a given stock symbol using yfinance.

    Input:
      - symbol (str): Ticker symbol of the stock
      - interval (str): Time interval (e.g. "1m", "5m", "15m")
      - period (str): Time span (e.g. "1d", "5d")
    Output:
      - pd.DataFrame: OHLCV with timestamps as index
    """
    df = yf.download(
        tickers=symbol,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {symbol} with interval={interval} and period={period}")

    df.dropna(inplace=True)
    df.index.name = "Timestamp"

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).strip().capitalize() for col in df.columns.values]
    else:
        df.columns = [col.capitalize() for col in df.columns]

    return df


def get_5min_data(symbol: str, days_back: int = 5) -> pd.DataFrame:
    """
    Retrieves 5-minute intraday data for the past N weekdays, market hours only (09:30–15:30).

    Input:
      - symbol (str): Ticker symbol
      - days_back (int): Number of past days to retrieve (default 5)
    Output:
      - pd.DataFrame: 5-minute OHLCV during market hours
    """
    df = yf.download(
        tickers=symbol,
        period=f"{days_back}d",
        interval="5m",
        progress=False,
        auto_adjust=False,
    )

    df = df[df.index.dayofweek < 5]
    df = df.between_time("09:30", "15:30")
    df.dropna(inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(col).strip().lower().replace(f"_{symbol.lower()}", "")
            for col in df.columns
        ]
    else:
        df.columns = [col.lower().replace(f"_{symbol.lower()}", "") for col in df.columns]

    df.index.name = "timestamp"

    return df
