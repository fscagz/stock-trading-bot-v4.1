# IBKR Data Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Alpaca's market data feed (SIP WebSocket + REST) with IBKR's TWS API via `ib_insync`, keeping Alpaca for all order execution.

**Architecture:** `ib_insync` connects to a locally-running IB Gateway process. Live bars arrive as 5-second OHLCV ticks via `reqRealTimeBars()`; a new `MinuteBarAggregator` class collects 12 ticks and emits one `Bar` — identical to what the rest of the bot already expects. Historical bars for backtesting come from `reqHistoricalData()`. The public interface of both `BarStream` and `BarFetcher` is unchanged; only their internals swap out.

**Tech Stack:** `ib_insync` (IBKR Python client), IB Gateway (local process, port 4001 live / 4002 paper), existing `Bar` dataclass, existing disk-cache JSON format.

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `bot/intraday/data/aggregator.py` | Collect 5-sec RealTimeBars per symbol, emit 1-min `Bar` to handler |
| Modify | `bot/intraday/data/stream.py` | Replace `StockDataStream` with `ib_insync` + `MinuteBarAggregator` |
| Modify | `bot/backtest/bar_fetcher.py` | Replace Alpaca REST with `ib_insync` `reqHistoricalData()` |
| Modify | `bot/live/runner.py` | Constructor gains `ibkr_host/port/client_id`; `run()` passes them to `BarStream` |
| Modify | `bot/live/__main__.py` | Read IBKR env vars; pass to `LiveRunner` |
| Modify | `bot/backtest/__main__.py` | Read IBKR env vars; update `BarFetcher` instantiation |
| Modify | `requirements.txt` | Add `ib_insync` |
| Modify | `.env` | Add `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID_STREAM`, `IBKR_CLIENT_ID_FETCHER` |
| Create | `tests/intraday/data/test_aggregator.py` | Unit tests for `MinuteBarAggregator` — fully testable without IBKR |

---

## Task 1: Add `ib_insync` dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add ib_insync to requirements.txt**

```text
# Core
alpaca-py
ib_insync
requests
python-dotenv
numpy
pandas
pytz
yfinance
fastapi
uvicorn

# Testing
pytest
```

- [ ] **Step 2: Install and verify**

```bash
cd /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v4.1
pip install ib_insync
python -c "from ib_insync import IB, Stock, util; print('ib_insync OK')"
```

Expected: `ib_insync OK`

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add ib_insync dependency for IBKR data migration"
```

---

## Task 2: Create MinuteBarAggregator

**Files:**
- Create: `bot/intraday/data/aggregator.py`
- Create: `tests/intraday/data/test_aggregator.py`

The aggregator's job: receive 5-second `RealTimeBar` objects from `ib_insync` and emit 1-minute `Bar` objects to the existing handler. It detects minute boundaries by comparing the truncated-to-minute timestamp of the incoming 5-sec bar against the previous bar's minute. When the minute rolls over, the completed buffer is emitted.

Note on `ib_insync` field name: `RealTimeBar.open_` has a trailing underscore (to avoid shadowing Python's `open` builtin). Historical `BarData.open` does not. The aggregator only deals with `RealTimeBar` so uses `open_`.

- [ ] **Step 1: Create the tests directory structure**

```bash
mkdir -p /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v4.1/tests/intraday/data
touch /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v4.1/tests/__init__.py
touch /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v4.1/tests/intraday/__init__.py
touch /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v4.1/tests/intraday/data/__init__.py
```

- [ ] **Step 2: Write the failing tests**

`tests/intraday/data/test_aggregator.py`:

```python
from __future__ import annotations
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import List

import pytest

from bot.intraday.data.aggregator import MinuteBarAggregator
from bot.intraday.types import Bar

_UTC = timezone.utc


def _make_5s(symbol: str, ts: datetime, o: float, h: float, l: float, c: float, vol: int):
    """Create a fake ib_insync RealTimeBar-like object."""
    return SimpleNamespace(
        time=ts,
        open_=o,
        high=h,
        low=l,
        close=c,
        volume=vol,
        contract=SimpleNamespace(symbol=symbol),
    )


def _ts(h: int, m: int, s: int) -> datetime:
    return datetime(2024, 1, 15, h, m, s, tzinfo=_UTC)


class TestMinuteBarAggregator:
    def test_no_emission_within_same_minute(self):
        emitted: List[Bar] = []
        agg = MinuteBarAggregator(emitted.append)

        agg.push("AAPL", _make_5s("AAPL", _ts(14, 30, 0), 150.0, 151.0, 149.5, 150.5, 1000))
        agg.push("AAPL", _make_5s("AAPL", _ts(14, 30, 5), 150.5, 152.0, 150.0, 151.0, 800))

        assert emitted == [], "should not emit mid-minute"

    def test_emits_on_minute_boundary(self):
        emitted: List[Bar] = []
        agg = MinuteBarAggregator(emitted.append)

        agg.push("AAPL", _make_5s("AAPL", _ts(14, 30, 0), 150.0, 151.0, 149.5, 150.5, 1000))
        agg.push("AAPL", _make_5s("AAPL", _ts(14, 30, 5), 150.5, 152.0, 150.0, 151.0,  800))
        agg.push("AAPL", _make_5s("AAPL", _ts(14, 31, 0), 151.0, 153.0, 150.8, 152.0,  600))

        assert len(emitted) == 1
        bar = emitted[0]
        assert bar.symbol == "AAPL"
        assert bar.open == pytest.approx(150.0)
        assert bar.high == pytest.approx(152.0)
        assert bar.low == pytest.approx(149.5)
        assert bar.close == pytest.approx(151.0)
        assert bar.volume == 1800
        assert bar.timestamp == _ts(14, 30, 0)

    def test_ohlcv_aggregation_correctness(self):
        emitted: List[Bar] = []
        agg = MinuteBarAggregator(emitted.append)

        # 3 bars in minute :30, then one bar in :31 to trigger emission
        agg.push("TSLA", _make_5s("TSLA", _ts(14, 30,  0), 200.0, 205.0, 199.0, 202.0, 500))
        agg.push("TSLA", _make_5s("TSLA", _ts(14, 30,  5), 202.0, 210.0, 201.0, 209.0, 700))
        agg.push("TSLA", _make_5s("TSLA", _ts(14, 30, 10), 209.0, 209.5, 195.0, 196.0, 300))
        agg.push("TSLA", _make_5s("TSLA", _ts(14, 31,  0), 196.0, 197.0, 195.5, 196.5, 200))

        assert len(emitted) == 1
        bar = emitted[0]
        assert bar.open == pytest.approx(200.0)   # first bar's open
        assert bar.high == pytest.approx(210.0)   # max high
        assert bar.low  == pytest.approx(195.0)   # min low
        assert bar.close == pytest.approx(196.0)  # last bar's close
        assert bar.volume == 1500                  # sum

    def test_independent_per_symbol(self):
        emitted: List[Bar] = []
        agg = MinuteBarAggregator(emitted.append)

        agg.push("AAPL", _make_5s("AAPL", _ts(14, 30, 0), 150.0, 151.0, 149.5, 150.5, 100))
        agg.push("TSLA", _make_5s("TSLA", _ts(14, 30, 0), 200.0, 201.0, 199.0, 200.5, 200))

        # Trigger AAPL minute boundary — should only emit AAPL bar
        agg.push("AAPL", _make_5s("AAPL", _ts(14, 31, 0), 150.5, 152.0, 150.0, 151.0, 50))
        assert len(emitted) == 1
        assert emitted[0].symbol == "AAPL"

        # Trigger TSLA minute boundary
        agg.push("TSLA", _make_5s("TSLA", _ts(14, 31, 0), 200.5, 202.0, 200.0, 201.0, 100))
        assert len(emitted) == 2
        assert emitted[1].symbol == "TSLA"

    def test_timestamp_is_minute_start(self):
        emitted: List[Bar] = []
        agg = MinuteBarAggregator(emitted.append)

        agg.push("AAPL", _make_5s("AAPL", _ts(14, 30, 5), 150.0, 151.0, 149.5, 150.5, 100))
        agg.push("AAPL", _make_5s("AAPL", _ts(14, 31, 0), 150.5, 151.0, 150.0, 150.8, 50))

        assert emitted[0].timestamp == _ts(14, 30, 0)  # truncated to minute start
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
cd /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v4.1
python -m pytest tests/intraday/data/test_aggregator.py -v
```

Expected: `ModuleNotFoundError: No module named 'bot.intraday.data.aggregator'`

- [ ] **Step 4: Implement MinuteBarAggregator**

`bot/intraday/data/aggregator.py`:

```python
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
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
python -m pytest tests/intraday/data/test_aggregator.py -v
```

Expected: 5 tests passing.

- [ ] **Step 6: Commit**

```bash
git add bot/intraday/data/aggregator.py tests/
git commit -m "feat: add MinuteBarAggregator — aggregates IBKR 5-sec bars to 1-min Bar objects"
```

---

## Task 3: Rewrite BarStream for IBKR

**Files:**
- Modify: `bot/intraday/data/stream.py`

The public interface is unchanged: `__init__`, `set_handler`, `subscribe`, `unsubscribe`, `run`, `run_with_reconnect`. Callers (including `LiveRunner`) see no difference except the constructor now takes `host/port/client_id` instead of `api_key/secret_key`.

`subscribe()` and `unsubscribe()` are called from the watchlist-refresh daemon thread while `run()` blocks the main thread. They use `ib.loop.call_soon_threadsafe()` to safely schedule work on the event loop thread.

The `stop_event` in `run_with_reconnect()` is wired to a monitor thread that calls `ib.disconnect()` when set — this causes `ib.run()` to return, which exits the loop.

- [ ] **Step 1: Replace stream.py**

`bot/intraday/data/stream.py`:

```python
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
            def _cancel():
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
```

- [ ] **Step 2: Smoke-test the import**

```bash
python -c "from bot.intraday.data.stream import BarStream; print('BarStream OK')"
```

Expected: `BarStream OK`

- [ ] **Step 3: Commit**

```bash
git add bot/intraday/data/stream.py
git commit -m "feat: rewrite BarStream to use IBKR reqRealTimeBars via ib_insync"
```

---

## Task 4: Rewrite BarFetcher for IBKR

**Files:**
- Modify: `bot/backtest/bar_fetcher.py`

The public interface stays the same: `__init__(...)`, `fetch(symbol, trade_date) -> List[Bar]`. The disk-cache format (`{symbol}_{date}.json` with `t/o/h/l/c/v` keys) is unchanged, so any bars already cached from previous Alpaca SIP runs are still valid and will be used as-is.

IBKR's historical data pacing limit is 60 requests per 10 minutes (~6/min). We enforce a 10-second minimum between non-cached requests. The connection is opened lazily on the first cache miss and reused across all `fetch()` calls. Call `close()` when done fetching.

`reqHistoricalData` returns `BarData` objects where `.date` is a `datetime` (ib_insync converts it). RTH=True restricts bars to regular session hours (9:30–16:00 ET), matching the existing Alpaca behavior.

- [ ] **Step 1: Replace bar_fetcher.py**

`bot/backtest/bar_fetcher.py`:

```python
from __future__ import annotations
import json
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

from bot.intraday.types import Bar

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
_PACING_INTERVAL = 10.0  # seconds between IBKR historical data requests

try:
    from ib_insync import IB, Stock
    _HAVE_IBKR = True
except ImportError:
    IB = None  # type: ignore[assignment,misc]
    _HAVE_IBKR = False


class BarFetcher:
    """Fetches 1-minute OHLCV bars from IBKR for a given symbol and date.

    Results are cached to disk as JSON. Existing cache files from prior Alpaca
    runs are compatible — same format, same filename convention.

    Connect lazily on first cache miss; reuse the connection across calls.
    Call close() when finished with a backtest run to release the connection.
    """

    def __init__(
        self,
        host: str,
        port: int,
        client_id: int,
        cache_dir: str = "backtest_results/cache",
    ) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ib: Optional[IB] = None
        self._last_request: float = 0.0

    def fetch(self, symbol: str, trade_date: date) -> List[Bar]:
        cache_path = self._cache_dir / f"{symbol}_{trade_date}.json"
        if cache_path.exists():
            raw = json.loads(cache_path.read_text())
            return [self._parse_bar(symbol, b) for b in raw]

        bars = self._fetch_from_ibkr(symbol, trade_date)
        if bars is not None:
            cache_path.write_text(json.dumps([self._bar_to_dict(b) for b in bars]))
        return bars or []

    def close(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()
        self._ib = None

    def _ensure_connected(self) -> None:
        if not _HAVE_IBKR:
            raise RuntimeError("ib_insync is required: pip install ib_insync")
        if self._ib is None or not self._ib.isConnected():
            self._ib = IB()
            self._ib.connect(self._host, self._port, clientId=self._client_id)
            logger.info("BarFetcher: connected to IB Gateway")

    def _fetch_from_ibkr(self, symbol: str, trade_date: date) -> Optional[List[Bar]]:
        self._ensure_connected()

        # Pace requests to stay within IBKR's 60-per-10-min limit
        elapsed = time.monotonic() - self._last_request
        if elapsed < _PACING_INTERVAL:
            time.sleep(_PACING_INTERVAL - elapsed)
        self._last_request = time.monotonic()

        contract = Stock(symbol, "SMART", "USD")
        end_dt = datetime(
            trade_date.year, trade_date.month, trade_date.day, 16, 0, tzinfo=_ET
        )

        try:
            ibkr_bars = self._ib.reqHistoricalData(
                contract,
                endDateTime=end_dt,
                durationStr="1 D",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=2,       # ib_insync returns datetime objects
                keepUpToDate=False,
            )
        except Exception as exc:
            logger.warning("BarFetcher: IBKR error for %s %s: %s", symbol, trade_date, exc)
            return None

        if not ibkr_bars:
            logger.debug("BarFetcher: no bars returned for %s %s", symbol, trade_date)
            return None

        bars: List[Bar] = []
        for b in ibkr_bars:
            ts = b.date
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            else:
                # Fallback: ib_insync occasionally returns a string
                ts = datetime.fromisoformat(str(ts)).replace(tzinfo=timezone.utc)
            bars.append(Bar(
                symbol=symbol,
                timestamp=ts,
                open=float(b.open),
                high=float(b.high),
                low=float(b.low),
                close=float(b.close),
                volume=int(b.volume),
            ))

        logger.debug("BarFetcher: %d bars fetched for %s %s", len(bars), symbol, trade_date)
        return bars

    def _parse_bar(self, symbol: str, b: dict) -> Bar:
        ts = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
        return Bar(
            symbol=symbol,
            timestamp=ts,
            open=float(b["o"]),
            high=float(b["h"]),
            low=float(b["l"]),
            close=float(b["c"]),
            volume=int(b["v"]),
        )

    def _bar_to_dict(self, bar: Bar) -> dict:
        return {
            "t": bar.timestamp.isoformat(),
            "o": bar.open,
            "h": bar.high,
            "l": bar.low,
            "c": bar.close,
            "v": bar.volume,
        }
```

- [ ] **Step 2: Smoke-test the import**

```bash
python -c "from bot.backtest.bar_fetcher import BarFetcher; print('BarFetcher OK')"
```

Expected: `BarFetcher OK`

- [ ] **Step 3: Commit**

```bash
git add bot/backtest/bar_fetcher.py
git commit -m "feat: rewrite BarFetcher to fetch 1-min bars from IBKR via ib_insync"
```

---

## Task 5: Wire IBKR params into entry points

**Files:**
- Modify: `bot/live/__main__.py`
- Modify: `bot/live/runner.py`
- Modify: `bot/backtest/__main__.py`

`LiveRunner` gains three IBKR constructor params. `__main__.py` files read them from environment variables. The `api_key`/`secret_key` params stay in `LiveRunner` because they're still needed for all broker calls (Alpaca).

- [ ] **Step 1: Update LiveRunner constructor in runner.py**

In `bot/live/runner.py`, update the `__init__` signature and store the IBKR params. The existing params (`api_key`, `secret_key`, etc.) are unchanged — just add three new ones at the front:

```python
class LiveRunner:
    def __init__(
        self,
        ibkr_host: str,
        ibkr_port: int,
        ibkr_client_id: int,
        api_key: str,
        secret_key: str,
        short_config: V4Config,
        long_config: V4Config,
        equity: float,
        etb_set: Set[str],
        risk_scale: float = 1.0,
    ) -> None:
        if risk_scale != 1.0:
            for cfg in (short_config, long_config):
                cfg.risk_per_trade = round(cfg.risk_per_trade * risk_scale, 6)
                cfg.max_portfolio_heat = min(round(cfg.max_portfolio_heat * risk_scale, 4), 1.0)
            logger.info(
                "Risk scale %.2f×: short risk_per_trade=%.4f long risk_per_trade=%.4f",
                risk_scale, short_config.risk_per_trade, long_config.risk_per_trade,
            )

        self._ibkr_host = ibkr_host
        self._ibkr_port = ibkr_port
        self._ibkr_client_id = ibkr_client_id
        self._short_cfg = short_config
        # ... rest of existing __init__ body unchanged from here ...
```

- [ ] **Step 2: Update BarStream instantiation inside runner.py run()**

Find the line in `run()` that creates `BarStream` (currently passes `api_key, secret_key`) and replace:

```python
# OLD:
self._stream = BarStream(self._api_key, self._secret_key, list(all_initial))

# NEW:
self._stream = BarStream(self._ibkr_host, self._ibkr_port, self._ibkr_client_id, list(all_initial))
```

- [ ] **Step 3: Update bot/live/__main__.py**

```python
"""Entry point for the V4 live short-momentum runner.

Usage:
    python -m bot.live
    python -m bot.live --risk-scale 0.5
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

import bot.broker_alpaca as broker
from bot.config import V4Config, make_long_config
from bot.live.runner import LiveRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 Momentum Short Live Runner")
    parser.add_argument(
        "--risk-scale", type=float, default=1.0,
        help="Scale risk_per_trade, max_position_pct, max_portfolio_heat (default 1.0)",
    )
    args = parser.parse_args()

    api_key = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]

    ibkr_host = os.getenv("IBKR_HOST", "127.0.0.1")
    ibkr_port = int(os.getenv("IBKR_PORT", "4001"))
    ibkr_client_id = int(os.getenv("IBKR_CLIENT_ID_STREAM", "1"))

    account = broker.get_account_info()
    equity = account["portfolio_value"]
    logger.info("Account equity: $%.2f | Status: %s", equity, account["status"])

    etb_set = broker.get_etb_set()
    logger.info("ETB set: %d shortable symbols", len(etb_set))

    runner = LiveRunner(
        ibkr_host=ibkr_host,
        ibkr_port=ibkr_port,
        ibkr_client_id=ibkr_client_id,
        api_key=api_key,
        secret_key=secret_key,
        short_config=V4Config(),
        long_config=make_long_config(),
        equity=equity,
        etb_set=etb_set,
        risk_scale=args.risk_scale,
    )
    runner.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Update bot/backtest/__main__.py BarFetcher instantiation**

Find the line in `bot/backtest/__main__.py` that creates `BarFetcher`:

```python
# After the existing load_dotenv and os.environ reads, add:
ibkr_host = os.getenv("IBKR_HOST", "127.0.0.1")
ibkr_port = int(os.getenv("IBKR_PORT", "4001"))
ibkr_client_id = int(os.getenv("IBKR_CLIENT_ID_FETCHER", "2"))

# Replace:
# fetcher = BarFetcher(api_key, secret_key)
# With:
fetcher = BarFetcher(ibkr_host, ibkr_port, ibkr_client_id)
```

Also wrap the backtest loop to close the fetcher when done. Find the section after `all_trades` is populated:

```python
# After the for d in days: loop, before metrics computation:
fetcher.close()
```

- [ ] **Step 5: Smoke-test entry point imports**

```bash
python -c "from bot.live.runner import LiveRunner; print('LiveRunner OK')"
python -c "from bot.backtest.__main__ import main; print('backtest __main__ OK')" 2>&1 | head -5
```

Expected: Both print OK (or the second shows its argparse help, not an import error).

- [ ] **Step 6: Commit**

```bash
git add bot/live/__main__.py bot/live/runner.py bot/backtest/__main__.py
git commit -m "feat: wire IBKR host/port/client_id into LiveRunner and backtest entry points"
```

---

## Task 6: Add IBKR environment variables

**Files:**
- Modify: `.env`

- [ ] **Step 1: Add IBKR vars to .env**

Open `.env` and append:

```bash
# IBKR IB Gateway connection (data feed)
# Port 4001 = live trading gateway, 4002 = paper trading gateway
IBKR_HOST=127.0.0.1
IBKR_PORT=4002
IBKR_CLIENT_ID_STREAM=1
IBKR_CLIENT_ID_FETCHER=2
```

Start with `IBKR_PORT=4002` (paper) until live trading is confirmed working.

- [ ] **Step 2: Full smoke-test (no IB Gateway required)**

```bash
python -c "
import os
os.environ.setdefault('IBKR_HOST', '127.0.0.1')
os.environ.setdefault('IBKR_PORT', '4002')
os.environ.setdefault('IBKR_CLIENT_ID_STREAM', '1')
os.environ.setdefault('IBKR_CLIENT_ID_FETCHER', '2')
from bot.intraday.data.stream import BarStream
from bot.backtest.bar_fetcher import BarFetcher
from bot.intraday.data.aggregator import MinuteBarAggregator
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass (aggregator unit tests + any existing tests).

- [ ] **Step 4: Commit**

```bash
git add .env
git commit -m "chore: add IBKR IB Gateway env vars (defaulting to paper port 4002)"
```

---

## Integration Testing (manual — requires running IB Gateway)

These steps require IB Gateway to be open and logged in.

- [ ] **Verify BarFetcher with a real symbol**

```python
# Run from project root: python -c "..."
from datetime import date
from bot.backtest.bar_fetcher import BarFetcher
f = BarFetcher("127.0.0.1", 4002, 2)
bars = f.fetch("AAPL", date(2024, 1, 15))
print(f"{len(bars)} bars fetched")
print(f"First: {bars[0]}")
print(f"Last:  {bars[-1]}")
f.close()
```

Expected: ~390 bars (one per regular-session minute), first at 09:30 ET, last at 15:59 ET.

- [ ] **Run a backtest over one week of dates**

```bash
python -m bot.backtest --start 2024-01-08 --end 2024-01-12 --long
```

Expected: Runs without errors. Results should be comparable to equivalent Alpaca SIP backtest (same signals, similar trade count — minor differences from bar boundary timing are normal).

- [ ] **Verify BarStream with paper trading**

Start IB Gateway on paper port (4002), then:

```bash
python -m bot.live --risk-scale 0.0
```

`--risk-scale 0.0` zeroes all position sizing so no orders fire. Watch logs for `BarStream: connected to IB Gateway` and bar arrival messages.

Expected: Bars arrive every ~5 seconds per subscribed symbol; 1-min Bar log entries appear at each minute boundary.
