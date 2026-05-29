from __future__ import annotations
from typing import Dict, Optional

from bot.intraday.types import Bar


class EMAIndicator:
    """EMA(period) using standard 2/(period+1) smoothing.
    Returns None until period bars have seeded the initial SMA."""

    def __init__(self, period: int) -> None:
        self.period = period
        self._multiplier = 2.0 / (period + 1)
        self._ema: Dict[str, float] = {}
        self._count: Dict[str, int] = {}
        self._seed_sum: Dict[str, float] = {}

    def update(self, bar: Bar) -> Optional[float]:
        sym = bar.symbol
        if sym not in self._count:
            self._count[sym] = 0
            self._seed_sum[sym] = 0.0

        self._count[sym] += 1

        if self._count[sym] < self.period:
            self._seed_sum[sym] += bar.close
            return None

        if self._count[sym] == self.period:
            self._seed_sum[sym] += bar.close
            self._ema[sym] = self._seed_sum[sym] / self.period
            return self._ema[sym]

        self._ema[sym] = bar.close * self._multiplier + self._ema[sym] * (1.0 - self._multiplier)
        return self._ema[sym]

    def get(self, symbol: str) -> Optional[float]:
        return self._ema.get(symbol)
