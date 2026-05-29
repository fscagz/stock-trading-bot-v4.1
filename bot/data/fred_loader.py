"""
FRED (Federal Reserve Economic Data) macro / regime data loader.

Provides: VIX (via yfinance ^VIX), 10Y-2Y yield spread, CPI, and other
macro series used as market context features.

FRED data is already point-in-time safe: each observation is recorded with
the date it was *published*, not the date it refers to.

Setup
-----
Set FRED_API_KEY in your .env (or environment) to use the fredapi library.
A free key is available at https://fred.stlouisfed.org/docs/api/api_key.html

Alternatively, set BOT_USE_FRED=false to skip FRED and use yfinance-only
proxies for VIX and yield curve.

Key Series
----------
VIXCLS      — CBOE Volatility Index (daily)
DGS10       — 10-Year Treasury Constant Maturity Rate (daily)
DGS2        — 2-Year Treasury Constant Maturity Rate (daily)
T10Y2Y      — 10Y-2Y Treasury Spread (daily, pre-computed by FRED)
CPIAUCSL    — CPI All Urban Consumers (monthly)
FEDFUNDS    — Federal Funds Effective Rate (monthly)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from fredapi import Fred
    _FREDAPI_AVAILABLE = True
except ImportError:
    _FREDAPI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_fred(api_key: Optional[str] = None) -> "Fred":
    if not _FREDAPI_AVAILABLE:
        raise ImportError("fredapi is not installed. Run: pip install fredapi")
    if api_key is None:
        from config import FRED_API_KEY
        api_key = FRED_API_KEY
    if not api_key:
        raise ValueError(
            "FRED API key is not set. "
            "Set FRED_API_KEY in your .env. "
            "Free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return Fred(api_key=api_key)


def _to_series(raw: pd.Series, name: str) -> pd.Series:
    """Normalise a FRED series: date index, float values, named."""
    s = raw.copy()
    s.index = pd.to_datetime(s.index).normalize()
    s = pd.to_numeric(s, errors="coerce")
    s.name = name
    return s.sort_index()


def _slice_as_of(series: pd.Series, as_of: date) -> pd.Series:
    """Return all observations on or before as_of (point-in-time safe)."""
    return series[series.index <= pd.Timestamp(as_of)]


# ---------------------------------------------------------------------------
# Individual series loaders
# ---------------------------------------------------------------------------

def load_vix(
    start: str,
    end: str,
    api_key: Optional[str] = None,
    prefer_fred: bool = True,
) -> pd.Series:
    """
    Load CBOE VIX daily closing levels.

    Tries FRED (VIXCLS) first if prefer_fred=True and API key is set,
    falls back to yfinance ^VIX.

    Parameters
    ----------
    start, end : str
        Date range 'YYYY-MM-DD'. end is inclusive.
    api_key : str, optional
    prefer_fred : bool

    Returns
    -------
    pd.Series — index=date, values=VIX level, name='vix'
    """
    if prefer_fred and api_key is not None:
        try:
            fred = _get_fred(api_key)
            raw = fred.get_series("VIXCLS", observation_start=start, observation_end=end)
            return _to_series(raw, "vix")
        except Exception:
            pass  # fall through to yfinance

    # yfinance fallback
    # yfinance end is exclusive — add one day
    end_yf = (pd.Timestamp(end) + timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download("^VIX", start=start, end=end_yf, interval="1d",
                     progress=False, auto_adjust=False)
    if df.empty:
        return pd.Series(dtype=float, name="vix")
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)  # drop ticker level, keep price names
    close = df["Close"] if "Close" in df.columns else df["close"]
    close.index = pd.to_datetime(close.index).normalize()
    close.name = "vix"
    return close.sort_index()


def load_yield_spread(
    start: str,
    end: str,
    api_key: Optional[str] = None,
) -> pd.Series:
    """
    Load 10Y-2Y Treasury yield spread (daily).

    Uses FRED series T10Y2Y (pre-computed spread). Falls back to computing
    DGS10 - DGS2 if T10Y2Y is unavailable.

    Returns
    -------
    pd.Series — index=date, values=spread in percentage points, name='yield_spread_10y2y'
    """
    fred = _get_fred(api_key)
    try:
        raw = fred.get_series("T10Y2Y", observation_start=start, observation_end=end)
        return _to_series(raw, "yield_spread_10y2y")
    except Exception:
        pass

    # Fallback: compute from constituent series
    dgs10 = fred.get_series("DGS10", observation_start=start, observation_end=end)
    dgs2 = fred.get_series("DGS2", observation_start=start, observation_end=end)
    s10 = _to_series(dgs10, "dgs10")
    s2 = _to_series(dgs2, "dgs2")
    spread = (s10 - s2).rename("yield_spread_10y2y")
    return spread.dropna()


def load_treasury_10y(
    start: str,
    end: str,
    api_key: Optional[str] = None,
) -> pd.Series:
    """10-Year Treasury constant maturity rate (%)."""
    fred = _get_fred(api_key)
    raw = fred.get_series("DGS10", observation_start=start, observation_end=end)
    return _to_series(raw, "treasury_10y")


def load_cpi(
    start: str,
    end: str,
    api_key: Optional[str] = None,
) -> pd.Series:
    """
    CPI All Urban Consumers (monthly, not seasonally adjusted).
    Returns YoY percentage change.
    """
    fred = _get_fred(api_key)
    raw = fred.get_series("CPIAUCSL", observation_start=start, observation_end=end)
    cpi = _to_series(raw, "cpi")
    cpi_yoy = cpi.pct_change(12).rename("cpi_yoy")
    return cpi_yoy.dropna()


def load_fed_funds(
    start: str,
    end: str,
    api_key: Optional[str] = None,
) -> pd.Series:
    """Federal Funds Effective Rate (monthly %)."""
    fred = _get_fred(api_key)
    raw = fred.get_series("FEDFUNDS", observation_start=start, observation_end=end)
    return _to_series(raw, "fed_funds")


# ---------------------------------------------------------------------------
# Composite macro context (used by features/market_context.py)
# ---------------------------------------------------------------------------

def get_macro_context_as_of(
    as_of: date,
    lookback_days: int = 252,
    api_key: Optional[str] = None,
    prefer_fred: bool = True,
) -> dict[str, Optional[float]]:
    """
    Return a snapshot of macro regime features as of `as_of`.

    All series are sliced to observations on or before `as_of` so that
    no future data leaks into the feature vector.

    Parameters
    ----------
    as_of : date
        Rebalance date.
    lookback_days : int
        How far back to fetch data for percentile / trend computations.
    api_key : str, optional
        FRED API key. Defaults to config.FRED_API_KEY.
    prefer_fred : bool
        Use FRED for VIX if key available; fallback to yfinance.

    Returns
    -------
    dict with keys:
        vix_level          — most recent VIX closing level
        vix_percentile     — VIX percentile vs trailing lookback_days
        yield_spread_10y2y — most recent 10Y-2Y spread (pp)
        yield_spread_trend — spread change over 63 days (positive = steepening)
        cpi_yoy            — most recent CPI YoY %
    All values are float or None if data unavailable.
    """
    start = (as_of - timedelta(days=lookback_days + 60)).strftime("%Y-%m-%d")
    end = as_of.strftime("%Y-%m-%d")
    out: dict[str, Optional[float]] = {
        "vix_level": None,
        "vix_percentile": None,
        "yield_spread_10y2y": None,
        "yield_spread_trend": None,
        "cpi_yoy": None,
    }

    # ---- VIX ----
    try:
        vix = load_vix(start, end, api_key=api_key, prefer_fred=prefer_fred)
        vix = _slice_as_of(vix, as_of)
        if not vix.empty:
            current_vix = float(vix.iloc[-1])
            out["vix_level"] = current_vix
            window = vix.iloc[-lookback_days:]
            out["vix_percentile"] = float(np.clip((window < current_vix).mean(), 0.0, 1.0))
    except Exception:
        pass

    # ---- Yield curve ----
    try:
        spread = load_yield_spread(start, end, api_key=api_key)
        spread = _slice_as_of(spread, as_of)
        if not spread.empty:
            out["yield_spread_10y2y"] = float(spread.iloc[-1])
            if len(spread) >= 63:
                out["yield_spread_trend"] = float(spread.iloc[-1] - spread.iloc[-63])
    except Exception:
        pass

    # ---- CPI ----
    try:
        cpi = load_cpi(start, end, api_key=api_key)
        cpi = _slice_as_of(cpi, as_of)
        if not cpi.empty:
            out["cpi_yoy"] = float(cpi.iloc[-1])
    except Exception:
        pass

    return out
