"""
Daily OHLCV data loader for the systematic pipeline.

Uses Yahoo Finance. Single-symbol and batch loading with a consistent
daily bar format (open, high, low, close, volume) and date index.
Aligned with the plan's "daily data" and 1–3 month forward return horizon.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf
from typing import List, Optional

# Standard column names (lowercase) for pipeline consistency
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _normalize_daily_df(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    """Ensure DataFrame has DatetimeIndex and lowercase ohlcv columns."""
    if df.empty:
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    # Unify column names to lowercase
    col_map = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=col_map)
    # Use close (prefer unadjusted); drop adj close
    if "adj close" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"adj close": "close"})
    elif "adj close" in df.columns:
        df = df.drop(columns=["adj close"])
    # Keep only OHLCV
    kept = [c for c in OHLCV_COLS if c in df.columns]
    df = df[kept].dropna(how="all")
    df.index.name = "date"
    return df


def get_daily(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = "2y",
) -> pd.DataFrame:
    """
    Load daily OHLCV for a single symbol.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. "AAPL").
    start : str, optional
        Start date (YYYY-MM-DD). Ignored if period is set.
    end : str, optional
        End date (YYYY-MM-DD). Ignored if period is set.
    period : str, optional
        yfinance period: "1mo", "3mo", "6mo", "1y", "2y", "5y", "max".
        Used when start/end not provided. Default "2y".

    Returns
    -------
    pd.DataFrame
        Index: date (normalized, no TZ). Columns: open, high, low, close, volume.
    """
    if start and end:
        df = yf.download(
            tickers=symbol,
            start=start,
            end=end,
            interval="1d",
            progress=False,
            auto_adjust=False,
            group_by="ticker",
        )
    else:
        df = yf.download(
            tickers=symbol,
            period=period or "2y",
            interval="1d",
            progress=False,
            auto_adjust=False,
            group_by="ticker",
        )
    if df.empty:
        return pd.DataFrame(columns=OHLCV_COLS)
    # Single ticker can come back with plain columns or MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df = df[symbol].copy() if symbol in df.columns.get_level_values(0) else df.droplevel(0, axis=1)
    return _normalize_daily_df(df, symbol)


def get_daily_batch(
    symbols: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = "2y",
) -> dict[str, pd.DataFrame]:
    """
    Load daily OHLCV for multiple symbols.

    Parameters
    ----------
    symbols : list of str
        Ticker symbols.
    start, end, period : optional
        Same as get_daily. Either (start, end) or period.

    Returns
    -------
    dict[str, pd.DataFrame]
        Map of symbol -> DataFrame with columns open, high, low, close, volume.
        Symbols with no data are omitted from the dict.
    """
    if not symbols:
        return {}
    if start and end:
        raw = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            interval="1d",
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )
    else:
        raw = yf.download(
            tickers=symbols,
            period=period or "2y",
            interval="1d",
            progress=False,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
        )
    if raw.empty:
        return {}
    result = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for sym in symbols:
            if sym in raw.columns.get_level_values(0):
                df = _normalize_daily_df(raw[sym].copy(), sym)
                if not df.empty:
                    result[sym] = df
    else:
        # Single symbol: yfinance returns plain columns
        sym = symbols[0]
        df = _normalize_daily_df(raw.copy(), sym)
        if not df.empty:
            result[sym] = df
    return result
