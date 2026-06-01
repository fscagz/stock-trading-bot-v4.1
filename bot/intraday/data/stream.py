from __future__ import annotations
import logging
import threading
from datetime import timezone
from typing import Callable, List, Optional, Set

from bot.intraday.types import Bar

logger = logging.getLogger(__name__)

BarHandler = Callable[[Bar], None]

try:
    from alpaca.data.live import StockDataStream
except ImportError:
    StockDataStream = None  # type: ignore[assignment,misc]


class BarStream:
    """Subscribes to Alpaca real-time 1-min bars via WebSocket.

    Usage:
        stream = BarStream(api_key, secret_key, symbols)
        stream.set_handler(my_handler)
        stream.run()   # blocks; run in a thread

    Call subscribe/unsubscribe while running to add/remove symbols dynamically.
    """

    def __init__(self, api_key: str, secret_key: str, symbols: List[str]) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._symbols: Set[str] = set(symbols)
        self._handler: Optional[BarHandler] = None
        self._client = None

    @property
    def symbols(self) -> Set[str]:
        return self._symbols

    def set_handler(self, handler: BarHandler) -> None:
        self._handler = handler

    def subscribe(self, symbol: str) -> None:
        if symbol in self._symbols:
            return
        self._symbols.add(symbol)
        if self._client is not None:
            self._client.subscribe_bars(self._make_on_bar(), symbol)

    def unsubscribe(self, symbol: str) -> None:
        self._symbols.discard(symbol)
        if self._client is not None:
            try:
                self._client.unsubscribe_bars(symbol)
            except Exception as exc:
                logger.warning("Unsubscribe failed for %s: %s", symbol, exc)

    def _make_on_bar(self) -> Callable:
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
        return _on_bar

    def run(self) -> None:
        if StockDataStream is None:
            raise RuntimeError("alpaca-py is required: pip install alpaca-py")
        self._client = StockDataStream(self._api_key, self._secret_key)
        on_bar = self._make_on_bar()
        if self._symbols:
            self._client.subscribe_bars(on_bar, *self._symbols)
        logger.info("BarStream starting for %d symbols", len(self._symbols))
        self._client.run()

    def run_with_reconnect(
        self,
        stop_event: threading.Event,
        max_retries: int = 20,
    ) -> None:
        """Block until stop_event is set, reconnecting on any websocket drop.

        Back-off: 10s, 20s, 40s, 80s … capped at 5 minutes.
        Gives up after max_retries *consecutive* failures (counter is not reset
        between attempts — only resets if the stream exits cleanly).
        """
        if StockDataStream is None:
            raise RuntimeError("alpaca-py is required: pip install alpaca-py")

        attempt = 0
        while not stop_event.is_set():
            try:
                self.run()
                logger.info("BarStream closed gracefully")
                return
            except Exception as exc:
                if stop_event.is_set():
                    return
                attempt += 1
                if attempt > max_retries:
                    logger.error(
                        "BarStream: max reconnect attempts (%d) reached, giving up",
                        max_retries,
                    )
                    return
                delay = min(10 * 2 ** (attempt - 1), 300)
                logger.warning(
                    "BarStream disconnected (attempt %d/%d): %s — reconnecting in %.0fs",
                    attempt, max_retries, exc, delay,
                )
                stop_event.wait(timeout=delay)
