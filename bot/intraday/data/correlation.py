"""
Pairwise return correlation for portfolio risk management.

Computes a 60-day daily return correlation matrix from yfinance and returns it
as a nested dict for O(1) lookup by (ticker_a, ticker_b).

Called once at session start; passed to IntradayBot to enforce the
max_position_correlation cap in _try_entry().
"""
from __future__ import annotations
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def compute_correlation_matrix(
    symbols: List[str],
    lookback_days: int = 60,
) -> Dict[str, Dict[str, float]]:
    """
    Return {ticker: {other_ticker: correlation}} for all symbol pairs.

    Uses daily close returns from yfinance. Pairs with fewer than 30
    overlapping observations get correlation 0.0 (treated as uncorrelated).

    Falls back to empty dict (no correlation enforcement) on any failure.
    """
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        logger.warning("yfinance unavailable; correlation cap will not be enforced")
        return {}

    logger.info("Computing correlation matrix for %d symbols...", len(symbols))
    try:
        raw = yf.download(
            symbols,
            period=f"{lookback_days}d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.warning("Correlation download failed: %s; cap not enforced", exc)
        return {}

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": symbols[0]})

    returns = close.pct_change().dropna(how="all")
    corr = returns.corr(min_periods=30).fillna(0.0)

    result: Dict[str, Dict[str, float]] = {}
    for sym in corr.columns:
        result[sym] = corr[sym].to_dict()

    logger.info("Correlation matrix computed (%d × %d)", len(result), len(result))
    return result
