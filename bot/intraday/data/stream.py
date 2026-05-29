from __future__ import annotations
import logging
from datetime import timezone
from typing import Callable, List, Optional

from bot.intraday.types import Bar

logger = logging.getLogger(__name__)

BarHandler = Callable[[Bar], None]


class BarStream:
    """Subscribes to Alpaca real-time 1-min bars via WebSocket.

    Usage:
        stream = BarStream(api_key, secret_key, symbols)
        stream.set_handler(my_handler)
        stream.run()   # blocks; run in a thread

    The handler is called with a Bar on each incoming 1-min bar.
    """

    def __init__(self, api_key: str, secret_key: str, symbols: List[str]) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._symbols = symbols
        self._handler: Optional[BarHandler] = None

    def set_handler(self, handler: BarHandler) -> None:
        self._handler = handler

    def run(self) -> None:
        try:
            from alpaca.data.live import StockDataStream
        except ImportError:
            raise RuntimeError("alpaca-py is required: pip install alpaca-py")

        stream = StockDataStream(self._api_key, self._secret_key)

        async def _on_bar(data) -> None:
            ts = data.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            bar = Bar(
                symbol=data.symbol,
                timestamp=ts,
                open=float(data.open),
                high=float(data.high),
                low=float(data.low),
                close=float(data.close),
                volume=int(data.volume),
            )
            if self._handler:
                self._handler(bar)

        stream.subscribe_bars(_on_bar, *self._symbols)
        logger.info("BarStream starting for %d symbols", len(self._symbols))
        stream.run()
