"""
Fundamental data fetcher — LIVE / PAPER SIGNALS ONLY.

Uses yfinance (Ticker.info). Returns today's values with no filing-date
alignment, which means they are NOT safe for historical backtesting.

For backtesting use bot/data/simfin_loader.py, which tags every data point
with its SEC filing (publish) date and supports point-in-time queries.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import yfinance as yf

# Canonical output keys for the feature engine (plan: value, quality, growth, stability)
FUNDAMENTAL_KEYS = [
    # Value
    "pe_ratio",
    "ev_ebitda",
    "fcf_yield",
    # Quality
    "roe",
    "gross_margin",
    "debt_to_equity",
    # Growth
    "revenue_growth",
    "eps_growth",
    # Profitability stability (optional)
    "earnings_consistency",
]


def _safe_get(info: Dict[str, Any], key: str) -> Optional[float]:
    """Get a numeric value from info dict; return None if missing or invalid."""
    val = info.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def get_fundamentals(symbol: str) -> Dict[str, Optional[float]]:
    """
    Fetch fundamental metrics for one symbol from yfinance.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. AAPL).

    Returns
    -------
    dict[str, float | None]
        Canonical keys (pe_ratio, ev_ebitda, fcf_yield, roe, gross_margin,
        debt_to_equity, revenue_growth, eps_growth, earnings_consistency).
        Missing or invalid values are None.
    """
    result: Dict[str, Optional[float]] = {k: None for k in FUNDAMENTAL_KEYS}
    try:
        t = yf.Ticker(symbol)
        info = t.info
        if not info:
            return result

        # Value
        result["pe_ratio"] = _safe_get(info, "trailingPE") or _safe_get(info, "forwardPE")
        result["ev_ebitda"] = _safe_get(info, "enterpriseToEbitda")
        fcf = _safe_get(info, "freeCashflow")
        mcap = _safe_get(info, "marketCap")
        if fcf is not None and mcap is not None and mcap > 0:
            result["fcf_yield"] = fcf / mcap

        # Quality
        result["roe"] = _safe_get(info, "returnOnEquity")
        result["gross_margin"] = _safe_get(info, "grossMargins")
        result["debt_to_equity"] = _safe_get(info, "debtToEquity")

        # Growth (yfinance often gives as decimal, e.g. 0.15 = 15%)
        result["revenue_growth"] = _safe_get(info, "revenueGrowth")
        result["eps_growth"] = _safe_get(info, "earningsGrowth")

        # Profitability stability: optional; use earningsQuarterlyGrowth variance or leave None
        # yfinance doesn't expose a direct "earnings consistency"; we could derive from financials
        result["earnings_consistency"] = None

    except Exception:
        pass
    return result


def get_fundamentals_batch(
    symbols: List[str],
    verbose: bool = False,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Fetch fundamentals for multiple symbols. One request per symbol (yfinance has no bulk API).

    Parameters
    ----------
    symbols : list of str
        Ticker symbols.
    verbose : bool
        If True, print progress.

    Returns
    -------
    dict[str, dict]
        symbol -> get_fundamentals(symbol) result. Failed symbols are still included
        with all values None.
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for i, sym in enumerate(symbols):
        if verbose and (i + 1) % 50 == 0:
            print(f"[fundamentals] {i + 1}/{len(symbols)}")
        out[sym] = get_fundamentals(sym)
    return out
