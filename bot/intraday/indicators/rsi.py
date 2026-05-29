from __future__ import annotations
from collections import deque
from typing import Deque, Dict, Optional

from bot.intraday.types import Bar


class RSIIndicator:
    """RSI(period) on streaming close prices. Returns None until period+1 bars are seen."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._closes: Dict[str, Deque[float]] = {}

    def update(self, bar: Bar) -> Optional[float]:
        sym = bar.symbol
        if sym not in self._closes:
            self._closes[sym] = deque(maxlen=self.period + 1)
        self._closes[sym].append(bar.close)
        return self._compute(sym)

    def get(self, symbol: str) -> Optional[float]:
        return self._compute(symbol)

    def _compute(self, sym: str) -> Optional[float]:
        closes = list(self._closes.get(sym, []))
        if len(closes) < self.period + 1:
            return None
        changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        avg_gain = sum(c for c in changes if c > 0) / self.period
        avg_loss = sum(-c for c in changes if c < 0) / self.period
        if avg_loss == 0:
            return 100.0
        if avg_gain == 0:
            return 0.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
