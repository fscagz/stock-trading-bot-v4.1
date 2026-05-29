"""
Point-in-time dataset builder: no look-ahead.

For each rebalance date, returns only data that would have been available
at market open on that date (e.g. closes through the previous trading day).
"""

from __future__ import annotations

from typing import List, Optional, Union

import pandas as pd

from data.daily_loader import get_daily, get_daily_batch

# Map period string to pandas DateOffset for lookback
_PERIOD_OFFSET = {
    "1mo": pd.DateOffset(months=1),
    "3mo": pd.DateOffset(months=3),
    "6mo": pd.DateOffset(months=6),
    "1y": pd.DateOffset(years=1),
    "2y": pd.DateOffset(years=2),
    "5y": pd.DateOffset(years=5),
}


def _start_from_lookback(as_of: pd.Timestamp, lookback_period: str) -> str:
    """Compute start date from as_of and period string."""
    offset = _PERIOD_OFFSET.get(lookback_period, pd.DateOffset(years=2))
    start_ts = as_of - offset
    return start_ts.strftime("%Y-%m-%d")


def slice_daily_as_of(
    df: pd.DataFrame,
    as_of_date: Union[str, pd.Timestamp],
    include_as_of_date: bool = False,
) -> pd.DataFrame:
    """
    Slice daily DataFrame to rows available as of a given date (no look-ahead).

    Parameters
    ----------
    df : pd.DataFrame
        Daily data with DatetimeIndex (or date index).
    as_of_date : str or pd.Timestamp
        Cutoff date (e.g. rebalance date). Only rows strictly before this date
        are kept, unless include_as_of_date is True.
    include_as_of_date : bool
        If False (default), return rows with index < as_of_date (standard:
        do not use rebalance day's close). If True, return index <= as_of_date.

    Returns
    -------
    pd.DataFrame
        Sliced view. Empty if no rows pass the cut.
    """
    if df.empty:
        return df
    as_of = pd.Timestamp(as_of_date).normalize()
    if include_as_of_date:
        return df.loc[df.index <= as_of].copy()
    return df.loc[df.index < as_of].copy()


def get_daily_as_of(
    symbol: str,
    as_of_date: Union[str, pd.Timestamp],
    include_as_of_date: bool = False,
    lookback_period: str = "2y",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load daily OHLCV for one symbol as of a date (point-in-time, no look-ahead).

    Fetches data up to as_of_date, then slices so only information available
    at market open on as_of_date is returned.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    as_of_date : str or pd.Timestamp
        Rebalance / cutoff date.
    include_as_of_date : bool
        If False (default), exclude the bar on as_of_date (use prior close only).
    lookback_period : str
        yfinance period when start/end not set (e.g. "2y").
    start, end : str, optional
        If both set, used instead of period. end will be capped to as_of_date.

    Returns
    -------
    pd.DataFrame
        Daily OHLCV with index < as_of_date (or <= if include_as_of_date).
    """
    as_of = pd.Timestamp(as_of_date).normalize()
    end_str = as_of.strftime("%Y-%m-%d")
    if start and end:
        df = get_daily(symbol, start=start, end=end_str)
    else:
        start_str = _start_from_lookback(as_of, lookback_period)
        df = get_daily(symbol, start=start_str, end=end_str)
    return slice_daily_as_of(df, as_of_date, include_as_of_date=include_as_of_date)


def get_daily_batch_as_of(
    symbols: List[str],
    as_of_date: Union[str, pd.Timestamp],
    include_as_of_date: bool = False,
    lookback_period: str = "2y",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load daily OHLCV for multiple symbols as of a date (point-in-time).

    Parameters
    ----------
    symbols : list of str
        Ticker symbols.
    as_of_date, include_as_of_date, lookback_period, start, end : optional
        Same semantics as get_daily_as_of.

    Returns
    -------
    dict[str, pd.DataFrame]
        symbol -> sliced daily DataFrame (no look-ahead).
    """
    as_of = pd.Timestamp(as_of_date).normalize()
    end_str = as_of.strftime("%Y-%m-%d")
    if start and end:
        panel = get_daily_batch(symbols, start=start, end=end_str)
    else:
        start_str = _start_from_lookback(as_of, lookback_period)
        panel = get_daily_batch(symbols, start=start_str, end=end_str)
    return {
        sym: slice_daily_as_of(df, as_of_date, include_as_of_date=include_as_of_date)
        for sym, df in panel.items()
    }


def rebalance_dates(
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    freq: str = "ME",
) -> pd.DatetimeIndex:
    """
    Generate rebalance dates (e.g. month-end) between start and end.

    Parameters
    ----------
    start, end : str or pd.Timestamp
        Inclusive range.
    freq : str
        Pandas offset: "ME" (month-end), "QE" (quarter-end), "BME" (business month-end), etc.

    Returns
    -------
    pd.DatetimeIndex
        Sorted rebalance dates in [start, end].
    """
    start_ = pd.Timestamp(start).normalize()
    end_ = pd.Timestamp(end).normalize()
    dr = pd.date_range(start=start_, end=end_, freq=freq)
    return dr
