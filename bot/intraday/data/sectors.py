"""
Sector classification for portfolio risk tracking.

Fetches the GICS sector for each symbol from yfinance at session start.
Falls back to 'Unknown' on any error (network, missing key, etc.).

Called once from run_paper.py and passed to IntradayBot via sector_map argument.
"""
from __future__ import annotations
import logging
from typing import Dict, List

import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_sector_map(symbols: List[str]) -> Dict[str, str]:
    """
    Return {ticker: sector} for each symbol.

    Makes one yfinance .info call per symbol. Slow (~0.5s/symbol) but only
    runs once at session start. Result is passed to IntradayBot constructor.
    """
    result: Dict[str, str] = {}
    for sym in symbols:
        try:
            info = yf.Ticker(sym).info
            result[sym] = info.get("sector", "Unknown")
        except Exception as exc:
            logger.warning("Could not fetch sector for %s: %s", sym, exc)
            result[sym] = "Unknown"
    return result
