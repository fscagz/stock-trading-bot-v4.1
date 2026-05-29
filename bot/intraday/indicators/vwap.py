from __future__ import annotations
from datetime import date
from typing import Dict, Optional

from bot.intraday.types import Bar


class VWAPIndicator:
    """Rolling intraday VWAP. Resets at the start of each calendar day per symbol."""

    def __init__(self) -> None:
        self._cum_pv: Dict[str, float] = {}
        self._cum_vol: Dict[str, float] = {}
        self._last_date: Dict[str, date] = {}

    def update(self, bar: Bar) -> float:
        sym = bar.symbol
        bar_date = bar.timestamp.date()

        if self._last_date.get(sym) != bar_date:
            self._cum_pv[sym] = 0.0
            self._cum_vol[sym] = 0.0
            self._last_date[sym] = bar_date

        self._cum_pv[sym] += bar.typical_price * bar.volume
        self._cum_vol[sym] += bar.volume

        return self._cum_pv[sym] / self._cum_vol[sym]

    def get(self, symbol: str) -> Optional[float]:
        vol = self._cum_vol.get(symbol, 0.0)
        if vol == 0.0:
            return None
        return self._cum_pv[symbol] / vol
