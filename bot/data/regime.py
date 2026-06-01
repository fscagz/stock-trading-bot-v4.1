"""
Market regime detection using SPY's 20-day moving average.

Experiment E1 showed:
  - 2022 bear with regime skip: $37,827 vs baseline $253 (+$37,574)
  - 2025-26 bull with regime skip: $34,837 vs baseline $33,345 (+$1,492)

The filter works in BOTH regimes: it dramatically reduces bear-market losses
while barely affecting bull-market gains (the 3 skipped trades in 2025-26
were actually slight net losers).
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from functools import lru_cache
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_SPY_CACHE: Optional[pd.DataFrame] = None


def _load_spy(start: str = "2019-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Load SPY daily OHLCV from yfinance. Cached in process memory."""
    global _SPY_CACHE
    if _SPY_CACHE is not None:
        return _SPY_CACHE
    from bot.data.daily_loader import get_daily
    end_str = end or date.today().isoformat()
    df = get_daily("SPY", start=start, end=end_str)
    if df.empty:
        logger.warning("RegimeFilter: could not load SPY data — defaulting to uptrend")
    _SPY_CACHE = df
    return _SPY_CACHE


class RegimeFilter:
    """Checks whether the broad market is in an uptrend or downtrend.

    Uses SPY closing price vs its 20-day simple moving average.
    When SPY is BELOW its MA on the most recent trading day before trade_date,
    the market is considered to be in a downtrend and long entries should be
    skipped (or sized down).

    Usage:
        rf = RegimeFilter()
        if rf.is_uptrend(date(2022, 6, 15)):
            # allow long entries
    """

    def __init__(self, ma_period: int = 20) -> None:
        self._ma_period = ma_period

    def is_uptrend(self, trade_date: date) -> bool:
        """Return True if SPY closed above its MA on the last trading day before trade_date."""
        spy = _load_spy()
        if spy.empty:
            return True
        target_ts = pd.Timestamp(trade_date)
        past = spy[spy.index < target_ts]
        if len(past) < self._ma_period:
            return True
        ma = float(past["close"].tail(self._ma_period).mean())
        last_close = float(past.iloc[-1]["close"])
        return last_close >= ma

    def risk_scale(self, trade_date: date) -> float:
        """Return 1.0 in uptrend, 0.0 in downtrend (for use with Simulator day_risk_scale)."""
        return 1.0 if self.is_uptrend(trade_date) else 0.0

    @staticmethod
    def precompute(start: date, end: date, ma_period: int = 20) -> dict[date, bool]:
        """Pre-compute uptrend flags for a date range (avoids per-day lookup overhead)."""
        rf = RegimeFilter(ma_period)
        days = []
        d = start
        while d <= end:
            if d.weekday() < 5:
                days.append(d)
            d += timedelta(days=1)
        return {d: rf.is_uptrend(d) for d in days}
