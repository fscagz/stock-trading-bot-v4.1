from __future__ import annotations
import logging
import threading
from typing import Callable, Dict, List, Optional, Set, Tuple

from bot.intraday.data.aggregator import MinuteBarAggregator
from bot.intraday.types import Bar

logger = logging.getLogger(__name__)

BarHandler = Callable[[Bar], None]

try:
    from ib_insync import IB, Stock
    _HAVE_IBKR = True
except ImportError:
    IB = None  # type: ignore[assignment,misc]
    _HAVE_IBKR = False

_SMART = "SMART"
_USD = "USD"
_WHAT_TO_SHOW = "TRADES"


class BarStream:
    """Subscribes to IBKR real-time 5-second bars via IB Gateway and emits
    aggregated 1-minute Bar objects to the registered handler.

    Requires a locally running IB Gateway (port 4001 live, 4002 paper).

    Usage:
        stream = BarStream(host, port, client_id, symbols)
        stream.set_handler(my_handler)
        stream.run()   # blocks; run in a thread or main loop

    Call subscribe/unsubscribe while running to add/remove symbols dynamically.
    """

    def __init__(
        self,
        host: str,
        port: int,
        client_id: int,
        symbols: List[str],
    ) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self._symbols: Set[str] = set(symbols)
        self._handler: Optional[BarHandler] = None
        self._ib: Optional[IB] = None
        self._aggregator: Optional[MinuteBarAggregator] = None
        # symbol -> (RealTimeBarList, callback) — kept so we can unsubscribe cleanly
        self._bar_lists: Dict[str, Tuple] = {}

    @property
    def symbols(self) -> Set[str]:
        return self._symbols

    def set_handler(self, handler: BarHandler) -> None:
        self._handler = handler

    def subscribe(self, symbol: str) -> None:
        if symbol in self._symbols:
            return
        self._symbols.add(symbol)
        if self._ib is not None and self._ib.isConnected():
            self._ib.loop.call_soon_threadsafe(self._subscribe_ibkr, symbol)

    def unsubscribe(self, symbol: str) -> None:
        self._symbols.discard(symbol)
        if self._ib is not None and symbol in self._bar_lists:
            bar_list, cb = self._bar_lists.pop(symbol)
            def _cancel() -> None:
                bar_list.updateEvent -= cb
                self._ib.cancelRealTimeBars(bar_list)
            self._ib.loop.call_soon_threadsafe(_cancel)

    def _subscribe_ibkr(self, symbol: str) -> None:
        contract = Stock(symbol, _SMART, _USD)
        bar_list = self._ib.reqRealTimeBars(contract, 5, _WHAT_TO_SHOW, False)

        def on_update(bars, has_new_bar: bool) -> None:
            if has_new_bar and self._aggregator:
                self._aggregator.push(symbol, bars[-1])

        bar_list.updateEvent += on_update
        self._bar_lists[symbol] = (bar_list, on_update)
        logger.debug("IBKR: subscribed to real-time bars for %s", symbol)

    def run(self, stop_event: Optional[threading.Event] = None) -> None:
        if not _HAVE_IBKR:
            raise RuntimeError("ib_insync is required: pip install ib_insync")
        self._ib = IB()
        self._aggregator = MinuteBarAggregator(self._handler or (lambda b: None))
        self._ib.connect(self._host, self._port, clientId=self._client_id)
        for symbol in list(self._symbols):
            self._subscribe_ibkr(symbol)
        logger.info("BarStream: connected to IB Gateway, streaming %d symbols", len(self._symbols))

        if stop_event is not None:
            def _monitor() -> None:
                stop_event.wait()
                if self._ib and self._ib.isConnected():
                    self._ib.disconnect()
            threading.Thread(target=_monitor, daemon=True, name="ibkr-stop-monitor").start()

        self._ib.run()

    def run_with_reconnect(
        self,
        stop_event: threading.Event,
        max_retries: int = 20,
    ) -> None:
        """Block until stop_event is set, reconnecting on any IB Gateway drop.

        Back-off: 10s, 20s, 40s, 80s … capped at 5 minutes.
        Gives up after max_retries consecutive failures.
        """
        if not _HAVE_IBKR:
            raise RuntimeError("ib_insync is required: pip install ib_insync")

        attempt = 0
        while not stop_event.is_set():
            try:
                self.run(stop_event=stop_event)
                if stop_event.is_set():
                    return
                logger.info("BarStream: IB Gateway connection closed cleanly")
                return
            except Exception as exc:
                if stop_event.is_set():
                    return
                attempt += 1
                if attempt > max_retries:
                    logger.error(
                        "BarStream: max reconnect attempts (%d) reached, giving up", max_retries
                    )
                    return
                delay = min(10 * 2 ** (attempt - 1), 300)
                logger.warning(
                    "BarStream: disconnected (attempt %d/%d): %s — reconnecting in %.0fs",
                    attempt, max_retries, exc, delay,
                )
            finally:
                if self._ib is not None:
                    try:
                        self._ib.disconnect()
                    except Exception:
                        pass
                    self._ib = None
                self._bar_lists.clear()
            stop_event.wait(timeout=delay)
