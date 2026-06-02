from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, List

from bot.intraday.types import Bar

BarHandler = Callable[[Bar], None]


class MinuteBarAggregator:
    """Collects ib_insync 5-second RealTimeBars and emits 1-minute Bar objects.

    A completed minute is emitted when the first 5-second bar of the NEXT minute
    arrives for that symbol. The final minute of the session is never auto-emitted
    (no subsequent bar arrives to trigger it) — this matches existing backtest
    behavior where the last partial bar is similarly ignored.
    """

    def __init__(self, handler: BarHandler) -> None:
        self._handler = handler
        self._buffers: Dict[str, List] = defaultdict(list)
        self._minute_key: Dict[str, int] = {}  # symbol -> Unix minute timestamp

    def push(self, symbol: str, bar_5s) -> None:
        """Accept one 5-second RealTimeBar. bar_5s is an ib_insync RealTimeBar."""
        ts = bar_5s.time
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        minute_key = int(ts.replace(second=0, microsecond=0).timestamp())
        prev_key = self._minute_key.get(symbol)

        if prev_key is not None and minute_key != prev_key:
            self._emit(symbol, prev_key)

        self._minute_key[symbol] = minute_key
        self._buffers[symbol].append(bar_5s)

    def _emit(self, symbol: str, minute_key: int) -> None:
        bars = self._buffers.pop(symbol, [])
        if not bars:
            return
        ts = datetime.fromtimestamp(minute_key, tz=timezone.utc)
        bar = Bar(
            symbol=symbol,
            timestamp=ts,
            open=float(bars[0].open_),
            high=max(float(b.high) for b in bars),
            low=min(float(b.low) for b in bars),
            close=float(bars[-1].close),
            volume=sum(int(b.volume) for b in bars),
        )
        self._handler(bar)
