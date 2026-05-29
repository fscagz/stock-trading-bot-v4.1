"""
Cache/store for daily OHLCV and fundamental data to avoid repeated API calls.

Uses config.CACHE_DIR. Daily data: one CSV per symbol. Fundamentals: one JSON per symbol.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from config import CACHE_DIR
from data.daily_loader import OHLCV_COLS

DAILY_SUBDIR = "daily"
FUNDAMENTALS_SUBDIR = "fundamentals"


def _daily_dir() -> Path:
    p = CACHE_DIR / DAILY_SUBDIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def _fundamentals_dir() -> Path:
    p = CACHE_DIR / FUNDAMENTALS_SUBDIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_cached_daily_path(symbol: str) -> Path:
    """Path where daily OHLCV for symbol is stored."""
    return _daily_dir() / f"{symbol.upper()}.csv"


def get_cached_fundamentals_path(symbol: str) -> Path:
    """Path where fundamentals for symbol are stored."""
    return _fundamentals_dir() / f"{symbol.upper()}.json"


def save_daily(symbol: str, df: pd.DataFrame) -> None:
    """
    Save daily OHLCV DataFrame to cache. Overwrites existing file.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. AAPL).
    df : pd.DataFrame
        DataFrame with date index and columns open, high, low, close, volume.
    """
    if df.empty:
        return
    kept = [c for c in OHLCV_COLS if c in df.columns]
    if len(kept) != len(OHLCV_COLS):
        return
    path = get_cached_daily_path(symbol)
    df = df[OHLCV_COLS].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df.to_csv(path, index=True)


def load_daily(symbol: str) -> Optional[pd.DataFrame]:
    """
    Load daily OHLCV from cache. Returns None if file missing or unreadable.

    Returns
    -------
    pd.DataFrame with date index and OHLCV columns, or None.
    """
    path = get_cached_daily_path(symbol)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index).normalize()
        df.index.name = "date"
        for c in OHLCV_COLS:
            if c not in df.columns:
                return None
        return df[OHLCV_COLS]
    except Exception:
        return None


def save_fundamentals(symbol: str, data: Dict[str, Any]) -> None:
    """
    Save fundamental metrics dict to cache. Overwrites existing file.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    data : dict
        Key -> value (must be JSON-serializable; floats/None are fine).
    """
    path = get_cached_fundamentals_path(symbol)
    with open(path, "w") as f:
        json.dump(data, f, indent=0)


def load_fundamentals(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Load fundamentals from cache. Returns None if file missing or unreadable.
    """
    path = get_cached_fundamentals_path(symbol)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def cache_age_days(path: Path) -> Optional[float]:
    """
    Age of a cache file in days (since last modification). None if file does not exist.
    """
    import time as _time
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    return (_time.time() - mtime) / 86400


def is_daily_stale(symbol: str, max_age_days: float = 1.0) -> bool:
    """True if no cache or cache is older than max_age_days."""
    age = cache_age_days(get_cached_daily_path(symbol))
    return age is None or age > max_age_days


def is_fundamentals_stale(symbol: str, max_age_days: float = 7.0) -> bool:
    """True if no cache or cache is older than max_age_days."""
    age = cache_age_days(get_cached_fundamentals_path(symbol))
    return age is None or age > max_age_days
