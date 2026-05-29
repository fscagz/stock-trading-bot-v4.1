"""
Fetches historical 1-minute bars from Alpaca for backtesting.

Uses the free IEX feed by default. Switch to feed="sip" for consolidated
data (requires Alpaca paid plan).

Usage:
    bars = load_historical_bars(["AAPL", "MSFT"], start, end, api_key, secret_key)
    # bars is Dict[str, List[Bar]]
"""
from __future__ import annotations
import logging
from datetime import datetime
from typing import Dict, List

from bot.intraday.types import Bar

logger = logging.getLogger(__name__)


def load_historical_bars(
    symbols: List[str],
    start: datetime,
    end: datetime,
    api_key: str,
    secret_key: str,
    feed: str = "iex",
) -> Dict[str, List[Bar]]:
    """
    Fetch 1-minute OHLCV bars from Alpaca for the given symbols and date range.

    Parameters
    ----------
    symbols   : list of ticker symbols
    start     : UTC datetime for range start (market hours only — Alpaca filters)
    end       : UTC datetime for range end
    api_key   : Alpaca API key
    secret_key: Alpaca secret key
    feed      : "iex" (free) or "sip" (paid consolidated)

    Returns
    -------
    Dict mapping symbol → list of Bar objects sorted ascending by timestamp.
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        raise RuntimeError("alpaca-py required: pip install alpaca-py")

    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=feed,
    )

    bars_data = client.get_stock_bars(request)
    result: Dict[str, List[Bar]] = {}

    for sym, raw_bars in bars_data.items():
        result[sym] = sorted(
            [
                Bar(
                    symbol=sym,
                    timestamp=b.timestamp,
                    open=float(b.open),
                    high=float(b.high),
                    low=float(b.low),
                    close=float(b.close),
                    volume=float(b.volume),
                )
                for b in raw_bars
            ],
            key=lambda b: b.timestamp,
        )
        logger.info("Loaded %d bars for %s", len(result[sym]), sym)

    return result
