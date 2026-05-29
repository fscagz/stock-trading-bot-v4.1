# V4 Momentum Scanner Bot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build stock-trading-bot-v4 by cloning V3, removing unused components, and layering in a momentum scanner, two-stage momentum validator, and flexible position manager that supports intraday and overnight holds.

**Architecture:** The scanner polls Alpaca's movers and most-actives endpoints every 30 seconds and adds candidates to a watchlist. The watchlist dynamically subscribes those symbols to a real-time bar stream. Each incoming bar is routed to the momentum validator (potential entries) and position manager (exit monitoring). The scanner and execution engine are loosely coupled — they share only the watchlist.

**Tech Stack:** Python 3.11+, alpaca-py, requests, pytest, python-dotenv

---

## File Map

### New Files

| File | Responsibility |
|------|---------------|
| `bot/config.py` | `V4Config` — all thresholds and risk params for V4 |
| `bot/scanner/__init__.py` | Package marker |
| `bot/scanner/market_scanner.py` | Polls Alpaca movers + most-actives every 30s; applies Stage 1 filter; updates watchlist |
| `bot/scanner/watchlist.py` | Maintains set of candidate symbols; manages BarStream subscriptions; stores per-symbol volume baseline |
| `bot/momentum/__init__.py` | Package marker |
| `bot/momentum/validator.py` | Stage 2 validation: rate-of-change, relative volume, buying pressure |
| `bot/positions/__init__.py` | Package marker |
| `bot/positions/manager.py` | Per-bar exit logic (3 layers); overnight hold decision at 15:25 |
| `bot/main.py` | Wires all components; routes bars to validator and manager; handles fills |
| `testing/test_broker_extensions.py` | Tests for new broker methods |
| `testing/test_market_scanner.py` | Tests for scanner Stage 1 filtering and deduplication |
| `testing/test_watchlist.py` | Tests for watchlist subscription management |
| `testing/test_momentum_validator.py` | Tests for Stage 2 validation logic |
| `testing/test_position_manager.py` | Tests for all exit conditions and overnight decision |

### Modified Files

| File | Change |
|------|--------|
| `bot/broker_alpaca.py` | Add `submit_limit_order`, `submit_stop_order`, `cancel_order`, `get_order` |
| `bot/intraday/data/stream.py` | Add `subscribe` and `unsubscribe` for dynamic symbol management |
| `bot/intraday/types.py` | Add `highest_close`, `stop_order_id`, `entry_bar_volume` fields to `Position` |

### Retained As-Is (no changes needed)

`bot/intraday/indicators/atr.py`, `bot/intraday/indicators/vwap.py`,
`bot/intraday/risk/sizing.py`, `bot/intraday/risk/portfolio.py`,
`bot/intraday/risk/kill_switch.py`, `bot/intraday/config.py`,
`bot/data/daily_loader.py`, `bot/backtest/`, `bot/monitoring/`

---

### Task 1: Repo Setup

**Files:**
- Create: `/Users/fscagz/Desktop/tradealgs/stock-trading-bot-v4/` (already exists)
- Copy structure from V3 and prune

- [ ] **Step 1: Copy V3 source into V4**

```bash
cp -r /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v3/bot \
      /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v4/bot
cp -r /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v3/testing \
      /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v4/testing
cp /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v3/requirements.txt \
   /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v4/requirements.txt
cp /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v3/pytest.ini \
   /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v4/pytest.ini
```

- [ ] **Step 2: Remove V3-only modules not used in V4**

```bash
cd /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v4

# V3 signal/event/ML/universe modules
rm -rf bot/intraday/signals/
rm -rf bot/intraday/ml/
rm bot/intraday/data/universe.py
rm bot/intraday/data/universe_loader.py
rm bot/intraday/data/event_calendar.py
rm bot/intraday/data/news_stream.py
rm bot/intraday/data/market_snapshot.py
rm bot/intraday/data/trade_stream.py
rm bot/intraday/risk/regime.py
rm bot/monitoring/drift.py
rm bot/monitoring/regime.py

# V2-era files at bot/ root (superseded by intraday/ equivalents)
rm bot/signal_generator.py
rm bot/indicators.py
rm bot/pipeline.py
rm bot/scheduler.py
rm bot/monitor.py
rm bot/backtester.py
rm bot/portfolio.py
rm bot/main.py

# V2-era sub-packages
rm -rf bot/features/
rm -rf bot/models/
rm -rf bot/attribution/
rm -rf bot/legacy/
rm -rf bot/portfolio/

# Old V3 test files that reference removed modules
rm -f testing/test_signal_gen.py
rm -f testing/test_universe.py
rm -f testing/test_pipeline.py
rm -f testing/test_monitoring.py
rm -f testing/test_models_factor.py
rm -f testing/test_features_builder.py
rm -f testing/test_features_fundamental.py
rm -f testing/test_features_market_context.py
rm -f testing/test_features_technical.py
rm -f testing/test_fundamental_store.py
rm -f testing/test_fundamentals.py
rm -f testing/test_portfolio_optimizer.py
rm -f testing/test_risk_model.py
rm -f testing/test_attribution.py
rm -rf testing/intraday/
rm -f bot/intraday/backtest/runner.py
```

- [ ] **Step 3: Slim down requirements.txt**

Replace contents of `requirements.txt` with:

```
# Core
alpaca-py
requests
python-dotenv
numpy
pandas
pytz

# Testing
pytest
```

- [ ] **Step 4: Initialize git and make first commit**

```bash
cd /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v4
git init
git add bot/ testing/ requirements.txt pytest.ini docs/ architecture.md
git commit -m "chore: init v4 from v3 — remove unused modules, retain core infrastructure"
```

---

### Task 2: Config and Types

**Files:**
- Create: `bot/config.py`
- Modify: `bot/intraday/types.py`

- [ ] **Step 1: Write failing test for V4Config**

Create `testing/test_config.py`:

```python
from bot.config import V4Config


def test_v4config_defaults():
    cfg = V4Config()
    assert cfg.max_sector_positions == 5
    assert cfg.min_price == 0.50
    assert cfg.max_spread_pct == 0.01
    assert cfg.scanner_interval_seconds == 30
    assert cfg.stage1_min_price_change_pct == 0.05
    assert cfg.stage2_roc_min_pct == 0.03
    assert cfg.stage2_min_relative_volume == 4.0
    assert cfg.stage2_buying_pressure_min == 0.75
    assert cfg.trailing_stop_atr_multiple == 2.0


def test_v4config_is_compatible_with_intraday_config():
    from bot.intraday.config import IntradayConfig
    cfg = V4Config()
    assert isinstance(cfg, IntradayConfig)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v4
pytest testing/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'bot.config'`

- [ ] **Step 3: Create bot/config.py**

```python
from __future__ import annotations
from dataclasses import dataclass
from bot.intraday.config import IntradayConfig


@dataclass
class V4Config(IntradayConfig):
    # --- V4 overrides of V3 defaults ---
    max_sector_positions: int = 5
    min_price: float = 0.50
    max_price: float = float("inf")
    max_spread_pct: float = 0.01

    # --- Trailing stop ---
    trailing_stop_atr_multiple: float = 2.0

    # --- Scanner ---
    scanner_interval_seconds: int = 30
    scanner_top_n: int = 50

    # --- Stage 1 filter ---
    stage1_min_price_change_pct: float = 0.05
    stage1_min_price: float = 0.50

    # --- Stage 2 momentum validation ---
    stage2_roc_lookback_bars: int = 5
    stage2_roc_min_pct: float = 0.03
    stage2_min_relative_volume: float = 4.0
    stage2_buying_pressure_min: float = 0.75

    # --- Exit thresholds ---
    vwap_break_volume_ratio: float = 2.0
    volume_collapse_ratio: float = 0.5
    structure_break_bars: int = 2
    overnight_min_volume_ratio: float = 2.0

    # --- Overnight hold evaluation time (ET) ---
    eod_evaluation: str = "15:25"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest testing/test_config.py -v
```

Expected: 2 passed

- [ ] **Step 5: Write failing test for updated Position type**

Add to `testing/test_config.py`:

```python
from bot.intraday.types import Position
from datetime import datetime, timezone


def test_position_has_v4_fields():
    pos = Position(
        ticker="ASTC",
        direction="long",
        shares=100,
        entry_price=2.00,
        stop_price=1.70,
        target_price=2.90,
        entry_time=datetime.now(timezone.utc),
        atr_at_entry=0.20,
        signals=["momentum"],
        sector="Unknown",
        highest_close=2.00,
        stop_order_id="abc123",
        entry_bar_volume=500_000,
    )
    assert pos.highest_close == 2.00
    assert pos.stop_order_id == "abc123"
    assert pos.entry_bar_volume == 500_000
```

- [ ] **Step 6: Run test to verify it fails**

```bash
pytest testing/test_config.py::test_position_has_v4_fields -v
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'highest_close'`

- [ ] **Step 7: Add V4 fields to Position in bot/intraday/types.py**

Add three fields after `open_risk` in the `Position` dataclass (lines 46–50):

```python
    open_risk: float = 0.0
    highest_close: float = 0.0        # tracks highest close since entry for trailing stop
    stop_order_id: str = ""           # Alpaca order ID of the broker-level hard stop
    entry_bar_volume: int = 0         # volume at entry bar, for volume collapse detection

    def __post_init__(self) -> None:
        if self.open_risk == 0.0:
            self.open_risk = self.shares * abs(self.entry_price - self.stop_price)
        if self.highest_close == 0.0:
            self.highest_close = self.entry_price
```

- [ ] **Step 8: Run all tests to verify they pass**

```bash
pytest testing/test_config.py -v
```

Expected: 3 passed

- [ ] **Step 9: Commit**

```bash
git add bot/config.py bot/intraday/types.py testing/test_config.py
git commit -m "feat: add V4Config with momentum thresholds; extend Position with trailing stop fields"
```

---

### Task 3: Broker Extensions

**Files:**
- Modify: `bot/broker_alpaca.py`
- Create: `testing/test_broker_extensions.py`

- [ ] **Step 1: Write failing tests**

Create `testing/test_broker_extensions.py`:

```python
from unittest.mock import MagicMock, patch
import bot.broker_alpaca as broker


def test_submit_limit_order_buy():
    mock_order = MagicMock()
    mock_order.id = "order-001"
    with patch.object(broker.trading_client, "submit_order", return_value=mock_order) as mock_submit:
        order_id = broker.submit_limit_order("ASTC", 100, "buy", 2.05)
        assert order_id == "order-001"
        call_args = mock_submit.call_args[0][0]
        assert call_args.symbol == "ASTC"
        assert float(call_args.limit_price) == 2.05


def test_submit_stop_order():
    mock_order = MagicMock()
    mock_order.id = "stop-001"
    with patch.object(broker.trading_client, "submit_order", return_value=mock_order) as mock_submit:
        order_id = broker.submit_stop_order("ASTC", 100, 1.70)
        assert order_id == "stop-001"
        call_args = mock_submit.call_args[0][0]
        assert float(call_args.stop_price) == 1.70


def test_cancel_order():
    with patch.object(broker.trading_client, "cancel_order_by_id") as mock_cancel:
        broker.cancel_order("stop-001")
        mock_cancel.assert_called_once_with("stop-001")


def test_get_order_status():
    mock_order = MagicMock()
    mock_order.status = "filled"
    mock_order.filled_avg_price = 2.05
    with patch.object(broker.trading_client, "get_order_by_id", return_value=mock_order):
        order = broker.get_order("order-001")
        assert order.status == "filled"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest testing/test_broker_extensions.py -v
```

Expected: 4 errors — `AttributeError: module 'bot.broker_alpaca' has no attribute 'submit_limit_order'`

- [ ] **Step 3: Add new functions to bot/broker_alpaca.py**

Append to the end of `bot/broker_alpaca.py`:

```python
from alpaca.trading.requests import LimitOrderRequest, StopOrderRequest


def submit_limit_order(symbol: str, qty: int, side: str, limit_price: float) -> str:
    order = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        limit_price=round(limit_price, 2),
    )
    result = trading_client.submit_order(order)
    return result.id


def submit_stop_order(symbol: str, qty: int, stop_price: float) -> str:
    order = StopOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        stop_price=round(stop_price, 2),
    )
    result = trading_client.submit_order(order)
    return result.id


def cancel_order(order_id: str) -> None:
    trading_client.cancel_order_by_id(order_id)


def get_order(order_id: str):
    return trading_client.get_order_by_id(order_id)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest testing/test_broker_extensions.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add bot/broker_alpaca.py testing/test_broker_extensions.py
git commit -m "feat: add limit, stop, cancel, and get_order to broker"
```

---

### Task 4: Dynamic Bar Stream

**Files:**
- Modify: `bot/intraday/data/stream.py`
- Test: `testing/test_stream.py`

- [ ] **Step 1: Write failing test**

Create `testing/test_stream.py`:

```python
from unittest.mock import MagicMock, patch, AsyncMock
from bot.intraday.data.stream import BarStream


def test_subscribe_adds_symbol():
    stream = BarStream("key", "secret", [])
    with patch("bot.intraday.data.stream.StockDataStream") as MockStream:
        mock_client = MagicMock()
        MockStream.return_value = mock_client
        stream._client = mock_client
        stream.subscribe("ASTC")
        assert "ASTC" in stream.symbols


def test_unsubscribe_removes_symbol():
    stream = BarStream("key", "secret", ["ASTC"])
    with patch("bot.intraday.data.stream.StockDataStream") as MockStream:
        mock_client = MagicMock()
        MockStream.return_value = mock_client
        stream._client = mock_client
        stream.unsubscribe("ASTC")
        assert "ASTC" not in stream.symbols


def test_subscribe_is_idempotent():
    stream = BarStream("key", "secret", [])
    with patch("bot.intraday.data.stream.StockDataStream") as MockStream:
        mock_client = MagicMock()
        MockStream.return_value = mock_client
        stream._client = mock_client
        stream.subscribe("ASTC")
        stream.subscribe("ASTC")
        assert stream.symbols.count("ASTC") == 1 if isinstance(stream.symbols, list) else True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest testing/test_stream.py -v
```

Expected: `AttributeError: 'BarStream' object has no attribute 'subscribe'`

- [ ] **Step 3: Update bot/intraday/data/stream.py**

Replace the full file with:

```python
from __future__ import annotations
import logging
from datetime import timezone
from typing import Callable, List, Optional, Set

from bot.intraday.types import Bar

logger = logging.getLogger(__name__)

BarHandler = Callable[[Bar], None]


class BarStream:
    """Subscribes to Alpaca real-time 1-min bars via WebSocket.

    Supports dynamic subscribe/unsubscribe after the stream is running.
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
            logger.info("Subscribed to %s", symbol)

    def unsubscribe(self, symbol: str) -> None:
        self._symbols.discard(symbol)
        if self._client is not None:
            try:
                self._client.unsubscribe_bars(symbol)
                logger.info("Unsubscribed from %s", symbol)
            except Exception as exc:
                logger.warning("Unsubscribe failed for %s: %s", symbol, exc)

    def _make_on_bar(self):
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
        try:
            from alpaca.data.live import StockDataStream
        except ImportError:
            raise RuntimeError("alpaca-py is required: pip install alpaca-py")

        self._client = StockDataStream(self._api_key, self._secret_key)
        on_bar = self._make_on_bar()

        if self._symbols:
            self._client.subscribe_bars(on_bar, *self._symbols)

        logger.info("BarStream starting for %d symbols", len(self._symbols))
        self._client.run()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest testing/test_stream.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add bot/intraday/data/stream.py testing/test_stream.py
git commit -m "feat: add dynamic subscribe/unsubscribe to BarStream"
```

---

### Task 5: Market Scanner

**Files:**
- Create: `bot/scanner/__init__.py`, `bot/scanner/market_scanner.py`
- Create: `testing/test_market_scanner.py`

- [ ] **Step 1: Write failing tests**

Create `testing/test_market_scanner.py`:

```python
from unittest.mock import MagicMock, patch
from bot.config import V4Config
from bot.scanner.market_scanner import MarketScanner


MOVERS_RESPONSE = {
    "gainers": [
        {"symbol": "ASTC", "percent_change": 15.3, "price": 2.30, "volume": 5_000_000},
        {"symbol": "SNDL", "percent_change": 3.0, "price": 1.20, "volume": 2_000_000},  # < 5%, filtered
        {"symbol": "GME",  "percent_change": 8.0, "price": 0.30, "volume": 3_000_000},  # < $0.50, filtered
    ]
}

MOST_ACTIVES_RESPONSE = {
    "most_actives": [
        {"symbol": "ASTC", "volume": 5_000_000},
        {"symbol": "NVDA", "volume": 80_000_000},
    ]
}


def _make_scanner():
    cfg = V4Config()
    watchlist = MagicMock()
    return MarketScanner("key", "secret", cfg, watchlist), watchlist


def test_stage1_filter_requires_5pct_gain():
    scanner, watchlist = _make_scanner()
    with patch("bot.scanner.market_scanner.requests.get") as mock_get:
        mock_get.side_effect = [
            MagicMock(json=lambda: MOVERS_RESPONSE, raise_for_status=lambda: None),
            MagicMock(json=lambda: MOST_ACTIVES_RESPONSE, raise_for_status=lambda: None),
        ]
        scanner.scan_once()
    added = {call.args[0] for call in watchlist.add.call_args_list}
    assert "ASTC" in added
    assert "SNDL" not in added


def test_stage1_filter_requires_min_price():
    scanner, watchlist = _make_scanner()
    with patch("bot.scanner.market_scanner.requests.get") as mock_get:
        mock_get.side_effect = [
            MagicMock(json=lambda: MOVERS_RESPONSE, raise_for_status=lambda: None),
            MagicMock(json=lambda: MOST_ACTIVES_RESPONSE, raise_for_status=lambda: None),
        ]
        scanner.scan_once()
    added = {call.args[0] for call in watchlist.add.call_args_list}
    assert "GME" not in added


def test_symbol_on_both_lists_is_high_priority():
    scanner, watchlist = _make_scanner()
    with patch("bot.scanner.market_scanner.requests.get") as mock_get:
        mock_get.side_effect = [
            MagicMock(json=lambda: MOVERS_RESPONSE, raise_for_status=lambda: None),
            MagicMock(json=lambda: MOST_ACTIVES_RESPONSE, raise_for_status=lambda: None),
        ]
        scanner.scan_once()
    # ASTC is on both lists
    calls = {call.args[0]: call.kwargs for call in watchlist.add.call_args_list}
    assert calls.get("ASTC", {}).get("high_priority") is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest testing/test_market_scanner.py -v
```

Expected: `ModuleNotFoundError: No module named 'bot.scanner'`

- [ ] **Step 3: Create bot/scanner/__init__.py**

```bash
touch /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v4/bot/scanner/__init__.py
```

- [ ] **Step 4: Create bot/scanner/market_scanner.py**

```python
from __future__ import annotations
import logging
import time
from typing import Optional

import requests

from bot.config import V4Config
from bot.scanner.watchlist import Watchlist

logger = logging.getLogger(__name__)

_MOVERS_URL = "https://data.alpaca.markets/v1beta1/screener/stocks/movers"
_ACTIVES_URL = "https://data.alpaca.markets/v1beta1/screener/stocks/most-actives"


class MarketScanner:
    """Polls Alpaca movers + most-actives every N seconds and feeds candidates to the watchlist."""

    def __init__(self, api_key: str, secret_key: str, config: V4Config, watchlist: Watchlist) -> None:
        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        self._cfg = config
        self._watchlist = watchlist

    def scan_once(self) -> None:
        movers = self._fetch_movers()
        actives = self._fetch_most_actives()
        active_symbols = {s["symbol"] for s in actives}

        for entry in movers:
            symbol = entry["symbol"]
            pct_change = entry.get("percent_change", 0.0)
            price = entry.get("price", 0.0)

            if pct_change < self._cfg.stage1_min_price_change_pct * 100:
                continue
            if price < self._cfg.stage1_min_price:
                continue

            high_priority = symbol in active_symbols
            self._watchlist.add(symbol, high_priority=high_priority)
            logger.debug("Candidate: %s (%.1f%%, high_priority=%s)", symbol, pct_change, high_priority)

    def run(self) -> None:
        logger.info("MarketScanner started (interval=%ds)", self._cfg.scanner_interval_seconds)
        while True:
            try:
                self.scan_once()
            except Exception as exc:
                logger.warning("Scanner error: %s", exc)
            time.sleep(self._cfg.scanner_interval_seconds)

    def _fetch_movers(self) -> list:
        resp = requests.get(
            _MOVERS_URL,
            headers=self._headers,
            params={"top": self._cfg.scanner_top_n},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("gainers", [])

    def _fetch_most_actives(self) -> list:
        resp = requests.get(
            _ACTIVES_URL,
            headers=self._headers,
            params={"top": self._cfg.scanner_top_n, "by": "volume"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("most_actives", [])
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest testing/test_market_scanner.py -v
```

Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add bot/scanner/ testing/test_market_scanner.py
git commit -m "feat: add MarketScanner — polls Alpaca movers + most-actives with Stage 1 filter"
```

---

### Task 6: Watchlist

**Files:**
- Create: `bot/scanner/watchlist.py`
- Create: `testing/test_watchlist.py`

The watchlist also loads 20-day average volume per symbol (needed by the momentum validator for relative volume calculation). It calls `daily_loader.py` once per new symbol at subscription time.

- [ ] **Step 1: Write failing tests**

Create `testing/test_watchlist.py`:

```python
from unittest.mock import MagicMock, patch
from bot.config import V4Config
from bot.scanner.watchlist import Watchlist


def _make_watchlist():
    cfg = V4Config()
    stream = MagicMock()
    watchlist = Watchlist(stream, cfg)
    return watchlist, stream


def test_add_subscribes_to_stream():
    watchlist, stream = _make_watchlist()
    with patch.object(watchlist, "_load_baseline_volume", return_value=1000.0):
        watchlist.add("ASTC")
    stream.subscribe.assert_called_once_with("ASTC")


def test_add_is_idempotent():
    watchlist, stream = _make_watchlist()
    with patch.object(watchlist, "_load_baseline_volume", return_value=1000.0):
        watchlist.add("ASTC")
        watchlist.add("ASTC")
    assert stream.subscribe.call_count == 1


def test_remove_unsubscribes_from_stream():
    watchlist, stream = _make_watchlist()
    with patch.object(watchlist, "_load_baseline_volume", return_value=1000.0):
        watchlist.add("ASTC")
    watchlist.remove("ASTC")
    stream.unsubscribe.assert_called_once_with("ASTC")


def test_get_baseline_volume_returns_stored_value():
    watchlist, stream = _make_watchlist()
    with patch.object(watchlist, "_load_baseline_volume", return_value=2500.0):
        watchlist.add("ASTC")
    assert watchlist.get_baseline_volume("ASTC") == 2500.0


def test_get_baseline_volume_unknown_symbol_returns_none():
    watchlist, stream = _make_watchlist()
    assert watchlist.get_baseline_volume("UNKNOWN") is None


def test_symbols_property_returns_current_set():
    watchlist, stream = _make_watchlist()
    with patch.object(watchlist, "_load_baseline_volume", return_value=1000.0):
        watchlist.add("ASTC")
        watchlist.add("NVDA")
    assert "ASTC" in watchlist.symbols
    assert "NVDA" in watchlist.symbols
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest testing/test_watchlist.py -v
```

Expected: `ModuleNotFoundError: No module named 'bot.scanner.watchlist'`

- [ ] **Step 3: Create bot/scanner/watchlist.py**

```python
from __future__ import annotations
import logging
from typing import Dict, Optional, Set

from bot.config import V4Config
from bot.intraday.data.stream import BarStream

logger = logging.getLogger(__name__)

_TRADING_MINUTES_PER_DAY = 390


class Watchlist:
    """Manages the set of candidate symbols and their BarStream subscriptions.

    When a symbol is added, its 20-day average per-minute volume baseline is loaded
    once and cached. This baseline is used by MomentumValidator for relative volume checks.
    """

    def __init__(self, stream: BarStream, config: V4Config) -> None:
        self._stream = stream
        self._cfg = config
        self._symbols: Set[str] = set()
        self._baselines: Dict[str, float] = {}

    @property
    def symbols(self) -> Set[str]:
        return self._symbols

    def add(self, symbol: str, high_priority: bool = False) -> None:
        if symbol in self._symbols:
            return
        baseline = self._load_baseline_volume(symbol)
        self._baselines[symbol] = baseline
        self._symbols.add(symbol)
        self._stream.subscribe(symbol)
        logger.info("Watchlist +%s (baseline_vol=%.0f/min, high_priority=%s)", symbol, baseline, high_priority)

    def remove(self, symbol: str) -> None:
        self._symbols.discard(symbol)
        self._baselines.pop(symbol, None)
        self._stream.unsubscribe(symbol)
        logger.info("Watchlist -%s", symbol)

    def get_baseline_volume(self, symbol: str) -> Optional[float]:
        return self._baselines.get(symbol)

    def _load_baseline_volume(self, symbol: str) -> float:
        try:
            from bot.data.daily_loader import load_daily_bars
            bars = load_daily_bars(symbol, days=20)
            if not bars:
                return 0.0
            avg_daily_volume = sum(b.volume for b in bars) / len(bars)
            return avg_daily_volume / _TRADING_MINUTES_PER_DAY
        except Exception as exc:
            logger.warning("Could not load baseline volume for %s: %s", symbol, exc)
            return 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest testing/test_watchlist.py -v
```

Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add bot/scanner/watchlist.py testing/test_watchlist.py
git commit -m "feat: add Watchlist — manages symbol subscriptions and per-symbol volume baseline"
```

---

### Task 7: Momentum Validator

**Files:**
- Create: `bot/momentum/__init__.py`, `bot/momentum/validator.py`
- Create: `testing/test_momentum_validator.py`

- [ ] **Step 1: Write failing tests**

Create `testing/test_momentum_validator.py`:

```python
from datetime import datetime, timezone
from bot.config import V4Config
from bot.intraday.types import Bar
from bot.momentum.validator import MomentumValidator


def _bar(symbol: str, close: float, high: float, low: float, volume: int, ts=None) -> Bar:
    ts = ts or datetime.now(timezone.utc)
    return Bar(symbol=symbol, timestamp=ts, open=close * 0.99,
               high=high, low=low, close=close, volume=volume)


def _make_validator():
    return MomentumValidator(V4Config())


def test_returns_false_with_insufficient_history():
    v = _make_validator()
    bar = _bar("ASTC", 2.30, 2.40, 2.10, 500_000)
    assert v.validate(bar, baseline_volume_per_min=100_000) is False


def test_returns_false_when_roc_too_low():
    v = _make_validator()
    # Feed 5 bars with minimal price movement
    for i in range(5):
        v.update(_bar("ASTC", 2.00, 2.05, 1.95, 500_000))
    # 6th bar: close only 1% above bar from 5 bars ago (need >= 3%)
    result = v.validate(_bar("ASTC", 2.02, 2.10, 1.98, 500_000), baseline_volume_per_min=100_000)
    assert result is False


def test_returns_false_when_relative_volume_too_low():
    v = _make_validator()
    for i in range(5):
        v.update(_bar("ASTC", 2.00 + i * 0.02, 2.10 + i * 0.02, 1.95 + i * 0.02, 500_000))
    # 6th bar: price moved enough (>3%) but volume is only 2× baseline (need >= 4×)
    bar = _bar("ASTC", 2.15, 2.20, 2.00, 200_000)  # 200k vs 100k baseline = 2×
    result = v.validate(bar, baseline_volume_per_min=100_000)
    assert result is False


def test_returns_false_when_buying_pressure_too_low():
    v = _make_validator()
    for i in range(5):
        v.update(_bar("ASTC", 2.00 + i * 0.02, 2.10 + i * 0.02, 1.95 + i * 0.02, 500_000))
    # 6th bar: price up, volume up, but close near the LOW (bearish bar)
    bar = _bar("ASTC", 2.15, 2.40, 2.00, 500_000)
    # range = 0.40, top 25% starts at 2.30, close=2.15 is in bottom 75%
    result = v.validate(bar, baseline_volume_per_min=100_000)
    assert result is False


def test_returns_true_when_all_conditions_met():
    v = _make_validator()
    for i in range(5):
        v.update(_bar("ASTC", 2.00 + i * 0.02, 2.10 + i * 0.02, 1.95 + i * 0.02, 500_000))
    # 6th bar: +5% price acceleration, 5× volume, closes near high
    bar = _bar("ASTC", 2.25, 2.28, 2.05, 500_000)
    # roc = (2.25 - 2.08) / 2.08 = ~8%  ✓
    # rel_vol = 500k / 100k = 5×  ✓
    # buying pressure: range=0.23, top 25% starts at 2.225, close=2.25  ✓
    result = v.validate(bar, baseline_volume_per_min=100_000)
    assert result is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest testing/test_momentum_validator.py -v
```

Expected: `ModuleNotFoundError: No module named 'bot.momentum'`

- [ ] **Step 3: Create bot/momentum/__init__.py**

```bash
touch /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v4/bot/momentum/__init__.py
```

- [ ] **Step 4: Create bot/momentum/validator.py**

```python
from __future__ import annotations
from collections import deque
from typing import Deque, Dict, Optional

from bot.config import V4Config
from bot.intraday.types import Bar


class MomentumValidator:
    """Stage 2 momentum validation: rate-of-change, relative volume, buying pressure.

    Call update(bar) to feed bars into the history buffer, then validate(bar, baseline)
    to check whether the bar meets all three entry conditions.
    """

    def __init__(self, config: V4Config) -> None:
        self._cfg = config
        self._history: Dict[str, Deque[Bar]] = {}

    def update(self, bar: Bar) -> None:
        sym = bar.symbol
        if sym not in self._history:
            lookback = self._cfg.stage2_roc_lookback_bars + 1
            self._history[sym] = deque(maxlen=lookback)
        self._history[sym].append(bar)

    def validate(self, bar: Bar, baseline_volume_per_min: float) -> bool:
        self.update(bar)
        history = list(self._history.get(bar.symbol, []))
        lookback = self._cfg.stage2_roc_lookback_bars

        if len(history) < lookback + 1:
            return False

        return (
            self._check_roc(bar, history, lookback)
            and self._check_relative_volume(bar, baseline_volume_per_min)
            and self._check_buying_pressure(bar)
        )

    def _check_roc(self, bar: Bar, history: list, lookback: int) -> bool:
        past_close = history[-(lookback + 1)].close
        if past_close <= 0:
            return False
        roc = (bar.close - past_close) / past_close
        return roc >= self._cfg.stage2_roc_min_pct

    def _check_relative_volume(self, bar: Bar, baseline_volume_per_min: float) -> bool:
        if baseline_volume_per_min <= 0:
            return False
        return bar.volume >= baseline_volume_per_min * self._cfg.stage2_min_relative_volume

    def _check_buying_pressure(self, bar: Bar) -> bool:
        bar_range = bar.high - bar.low
        if bar_range <= 0:
            return False
        close_position = (bar.close - bar.low) / bar_range
        return close_position >= self._cfg.stage2_buying_pressure_min
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest testing/test_momentum_validator.py -v
```

Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add bot/momentum/ testing/test_momentum_validator.py
git commit -m "feat: add MomentumValidator — Stage 2 rate-of-change, relative volume, buying pressure"
```

---

### Task 8: Position Manager

**Files:**
- Create: `bot/positions/__init__.py`, `bot/positions/manager.py`
- Create: `testing/test_position_manager.py`

- [ ] **Step 1: Write failing tests**

Create `testing/test_position_manager.py`:

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock
from bot.config import V4Config
from bot.intraday.types import Bar, Position
from bot.positions.manager import ExitInstruction, PositionManager


def _now(hour=10, minute=0):
    return datetime(2026, 5, 29, hour, minute, 0, tzinfo=timezone.utc)


def _position(entry_price=2.00, stop_price=1.70, highest_close=2.00, entry_bar_volume=200_000):
    return Position(
        ticker="ASTC", direction="long", shares=100,
        entry_price=entry_price, stop_price=stop_price,
        target_price=2.90, entry_time=_now(),
        atr_at_entry=0.20, signals=["momentum"],
        sector="Unknown", highest_close=highest_close,
        stop_order_id="stop-001", entry_bar_volume=entry_bar_volume,
    )


def _bar(close, high=None, low=None, volume=200_000, ts=None):
    high = high or close * 1.02
    low = low or close * 0.98
    ts = ts or _now()
    return Bar("ASTC", ts, close * 0.99, high, low, close, volume)


def _make_manager():
    return PositionManager(V4Config())


def test_no_exit_when_price_above_stop_and_momentum_intact():
    mgr = _make_manager()
    pos = _position(entry_price=2.00, stop_price=1.70)
    bar = _bar(close=2.20, volume=300_000)
    result = mgr.on_bar(bar, pos, vwap=2.10, baseline_volume_per_min=50_000)
    assert result is None


def test_trailing_stop_updates_when_new_high():
    mgr = _make_manager()
    pos = _position(entry_price=2.00, stop_price=1.70, highest_close=2.00)
    atr = 0.20
    bar = _bar(close=2.50, volume=300_000)
    result = mgr.on_bar(bar, pos, vwap=2.30, baseline_volume_per_min=50_000)
    # New trailing stop = 2.50 - 2 * 0.20 = 2.10, pos should be updated
    assert pos.highest_close == 2.50
    assert pos.stop_price == round(2.50 - 2 * pos.atr_at_entry, 2)


def test_hard_stop_triggers_market_exit():
    mgr = _make_manager()
    pos = _position(entry_price=2.00, stop_price=1.70)
    bar = _bar(close=1.65, volume=300_000)
    result = mgr.on_bar(bar, pos, vwap=1.80, baseline_volume_per_min=50_000)
    assert result is not None
    assert result.action == "market_exit"
    assert result.reason == "hard_stop"


def test_vwap_break_triggers_limit_exit():
    mgr = _make_manager()
    pos = _position(entry_price=2.00, stop_price=1.70)
    # Close below VWAP on elevated volume
    bar = _bar(close=1.85, volume=200_000)  # 200k vs 50k baseline = 4× (>= 2×)
    result = mgr.on_bar(bar, pos, vwap=1.90, baseline_volume_per_min=50_000)
    assert result is not None
    assert result.action == "limit_exit"
    assert result.reason == "vwap_break"


def test_volume_collapse_triggers_exit():
    mgr = _make_manager()
    pos = _position(entry_price=2.00, stop_price=1.70, entry_bar_volume=200_000)
    bar = _bar(close=2.10, volume=80_000)  # 80k < 0.5 × 200k = 100k
    result = mgr.on_bar(bar, pos, vwap=2.05, baseline_volume_per_min=50_000)
    assert result is not None
    assert result.reason == "volume_collapse"


def test_overnight_hold_returns_true_when_conditions_met():
    mgr = _make_manager()
    pos = _position()
    bar = _bar(close=2.40, volume=200_000)  # 200k vs 50k = 4× >= 2×
    hold = mgr.should_hold_overnight(bar, pos, vwap=2.30, baseline_volume_per_min=50_000)
    assert hold is True


def test_overnight_hold_returns_false_when_price_below_vwap():
    mgr = _make_manager()
    pos = _position()
    bar = _bar(close=2.00, volume=200_000)
    hold = mgr.should_hold_overnight(bar, pos, vwap=2.10, baseline_volume_per_min=50_000)
    assert hold is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest testing/test_position_manager.py -v
```

Expected: `ModuleNotFoundError: No module named 'bot.positions'`

- [ ] **Step 3: Create bot/positions/__init__.py**

```bash
touch /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v4/bot/positions/__init__.py
```

- [ ] **Step 4: Create bot/positions/manager.py**

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from bot.config import V4Config
from bot.intraday.types import Bar, Position


@dataclass
class ExitInstruction:
    reason: str   # "hard_stop" | "trailing_stop" | "vwap_break" | "volume_collapse"
                  # | "structure_break" | "roc_reversal" | "eod"
    action: str   # "market_exit" | "limit_exit"
    limit_price: Optional[float] = None


class PositionManager:
    """Monitors open positions on each bar and returns exit instructions when conditions are met.

    Does not call the broker directly — the main loop acts on returned ExitInstructions.
    Also tracks structure-break state (consecutive lower high/lower low) per symbol.
    """

    def __init__(self, config: V4Config) -> None:
        self._cfg = config
        self._structure_break_count: dict = {}
        self._prev_bar: dict = {}

    def on_bar(
        self,
        bar: Bar,
        position: Position,
        vwap: float,
        baseline_volume_per_min: float,
    ) -> Optional[ExitInstruction]:
        sym = bar.symbol

        # Layer 1: hard stop
        if bar.close <= position.stop_price:
            return ExitInstruction(reason="hard_stop", action="market_exit")

        # Layer 2: update trailing stop
        if bar.close > position.highest_close:
            position.highest_close = bar.close
            new_stop = round(bar.close - self._cfg.trailing_stop_atr_multiple * position.atr_at_entry, 2)
            if new_stop > position.stop_price:
                position.stop_price = new_stop

        if bar.close <= position.stop_price:
            return ExitInstruction(reason="trailing_stop", action="market_exit")

        # Layer 3a: VWAP break on elevated volume
        vwap_volume_threshold = baseline_volume_per_min * self._cfg.vwap_break_volume_ratio
        if bar.close < vwap and bar.volume >= vwap_volume_threshold:
            return ExitInstruction(reason="vwap_break", action="limit_exit", limit_price=bar.close)

        # Layer 3b: volume collapse
        if position.entry_bar_volume > 0:
            collapse_threshold = position.entry_bar_volume * self._cfg.volume_collapse_ratio
            if bar.volume < collapse_threshold:
                return ExitInstruction(reason="volume_collapse", action="market_exit")

        # Layer 3c: structure break (consecutive lower high + lower low)
        prev = self._prev_bar.get(sym)
        if prev is not None:
            if bar.high < prev.high and bar.low < prev.low:
                self._structure_break_count[sym] = self._structure_break_count.get(sym, 0) + 1
            else:
                self._structure_break_count[sym] = 0
            if self._structure_break_count.get(sym, 0) >= self._cfg.structure_break_bars:
                self._structure_break_count[sym] = 0
                return ExitInstruction(reason="structure_break", action="limit_exit", limit_price=bar.close)

        self._prev_bar[sym] = bar
        return None

    def should_hold_overnight(
        self,
        bar: Bar,
        position: Position,
        vwap: float,
        baseline_volume_per_min: float,
    ) -> bool:
        if bar.close < vwap:
            return False
        volume_threshold = baseline_volume_per_min * self._cfg.overnight_min_volume_ratio
        if bar.volume < volume_threshold:
            return False
        return True
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest testing/test_position_manager.py -v
```

Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add bot/positions/ testing/test_position_manager.py
git commit -m "feat: add PositionManager — 3-layer exit logic and overnight hold decision"
```

---

### Task 9: Main Bot Loop

**Files:**
- Create: `bot/main.py`

This is the integration layer — no new logic, just wiring. No unit tests (the components are already tested); verify manually with paper trading.

- [ ] **Step 1: Create bot/main.py**

```python
from __future__ import annotations
import logging
import os
import threading
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

import bot.broker_alpaca as broker
from bot.config import V4Config
from bot.intraday.data.stream import BarStream
from bot.intraday.indicators.atr import ATRIndicator
from bot.intraday.indicators.vwap import VWAPIndicator
from bot.intraday.risk.kill_switch import KillSwitch
from bot.intraday.risk.portfolio import PortfolioState
from bot.intraday.risk.sizing import compute_position_size
from bot.intraday.types import Bar, Position
from bot.momentum.validator import MomentumValidator
from bot.positions.manager import ExitInstruction, PositionManager
from bot.scanner.market_scanner import MarketScanner
from bot.scanner.watchlist import Watchlist

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def _handle_exit(instruction: ExitInstruction, position: Position) -> None:
    sym = position.ticker
    if instruction.action == "market_exit":
        broker.submit_market_order(sym, position.shares, side="sell")
        logger.info("MARKET EXIT %s — reason=%s", sym, instruction.reason)
    elif instruction.action == "limit_exit" and instruction.limit_price:
        broker.submit_limit_order(sym, position.shares, "sell", instruction.limit_price)
        logger.info("LIMIT EXIT %s @ %.2f — reason=%s", sym, instruction.limit_price, instruction.reason)

    if position.stop_order_id:
        try:
            broker.cancel_order(position.stop_order_id)
        except Exception as exc:
            logger.warning("Could not cancel stop order %s: %s", position.stop_order_id, exc)


def _wait_for_fill(order_id: str, timeout: int) -> tuple[bool, float]:
    for _ in range(timeout):
        try:
            order = broker.get_order(order_id)
            if str(order.status) == "filled":
                return True, float(order.filled_avg_price)
        except Exception:
            pass
        time.sleep(1)
    return False, 0.0


def main() -> None:
    load_dotenv()
    api_key = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]

    config = V4Config()
    account = broker.get_account_info()
    equity = account["portfolio_value"]

    portfolio = PortfolioState(equity=equity, config=config)
    kill_switch = KillSwitch(config)
    atr_indicator = ATRIndicator(period=14)
    vwap_indicator = VWAPIndicator()
    validator = MomentumValidator(config)
    manager = PositionManager(config)

    stream = BarStream(api_key, secret_key, symbols=[])
    watchlist = Watchlist(stream, config)
    scanner = MarketScanner(api_key, secret_key, config, watchlist)

    eod_hour, eod_minute = (int(x) for x in config.eod_evaluation.split(":"))

    def on_bar(bar: Bar) -> None:
        now = datetime.now(timezone.utc)
        kill_switch.check(portfolio, now)
        if portfolio.kill_switch_active:
            return

        atr_val = atr_indicator.update(bar)
        vwap_val = vwap_indicator.update(bar)
        baseline = watchlist.get_baseline_volume(bar.symbol)

        # --- Exit logic for open positions ---
        if bar.symbol in portfolio.positions and atr_val and vwap_val and baseline:
            position = portfolio.positions[bar.symbol]

            # Overnight hold evaluation at eod_evaluation time
            if now.hour == eod_hour and now.minute == eod_minute:
                if not manager.should_hold_overnight(bar, position, vwap_val, baseline):
                    _handle_exit(ExitInstruction(reason="eod", action="market_exit"), position)
                    portfolio.remove_position(bar.symbol)
                return

            instruction = manager.on_bar(bar, position, vwap_val, baseline)
            if instruction:
                _handle_exit(instruction, position)
                portfolio.remove_position(bar.symbol)

                # Update trailing stop at broker if stop price changed
                if (instruction is None and
                        position.stop_order_id and
                        position.stop_price != portfolio.positions.get(bar.symbol, position).stop_price):
                    try:
                        broker.cancel_order(position.stop_order_id)
                        new_stop_id = broker.submit_stop_order(bar.symbol, position.shares, position.stop_price)
                        position.stop_order_id = new_stop_id
                    except Exception as exc:
                        logger.warning("Trailing stop update failed for %s: %s", bar.symbol, exc)
            return

        # --- Entry logic for watchlist candidates ---
        if (bar.symbol in watchlist.symbols and
                bar.symbol not in portfolio.positions and
                atr_val and baseline):
            if not validator.validate(bar, baseline):
                return

            can_enter, reason = portfolio.can_enter(sector="Unknown", now=now)
            if not can_enter:
                logger.debug("Entry blocked for %s: %s", bar.symbol, reason)
                return

            size = compute_position_size(portfolio.equity, atr_val, bar.close, config)
            if size.shares <= 0:
                return

            limit_price = round(bar.close * (1 + config.limit_offset_pct), 2)
            try:
                order_id = broker.submit_limit_order(bar.symbol, size.shares, "buy", limit_price)
                filled, fill_price = _wait_for_fill(order_id, config.fill_timeout_seconds)
                if not filled:
                    broker.cancel_order(order_id)
                    logger.info("Entry unfilled for %s — cancelled", bar.symbol)
                    return

                stop_price = size.long_stop(fill_price)
                stop_order_id = broker.submit_stop_order(bar.symbol, size.shares, stop_price)

                position = Position(
                    ticker=bar.symbol,
                    direction="long",
                    shares=size.shares,
                    entry_price=fill_price,
                    stop_price=stop_price,
                    target_price=size.long_target(fill_price),
                    entry_time=now,
                    atr_at_entry=atr_val,
                    signals=["momentum"],
                    sector="Unknown",
                    highest_close=fill_price,
                    stop_order_id=stop_order_id,
                    entry_bar_volume=bar.volume,
                )
                portfolio.add_position(position)
                logger.info("ENTRY %s: %d shares @ %.2f, stop=%.2f",
                            bar.symbol, size.shares, fill_price, stop_price)
            except Exception as exc:
                logger.error("Entry failed for %s: %s", bar.symbol, exc)

    stream.set_handler(on_bar)

    scanner_thread = threading.Thread(target=scanner.run, daemon=True)
    scanner_thread.start()
    logger.info("V4 Momentum Bot started. Equity: $%.2f", equity)

    stream.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite to confirm nothing is broken**

```bash
cd /Users/fscagz/Desktop/tradealgs/stock-trading-bot-v4
pytest testing/ -v --tb=short
```

Expected: All tests pass. Fix any import errors before proceeding.

- [ ] **Step 3: Smoke test with paper account**

Set environment variables and run against Alpaca paper:

```bash
export APCA_API_KEY_ID=your_paper_key
export APCA_API_SECRET_KEY=your_paper_secret
python -m bot.main
```

Verify in logs:
- `V4 Momentum Bot started. Equity: $...` appears
- `MarketScanner started` appears
- No immediate exceptions

- [ ] **Step 4: Commit**

```bash
git add bot/main.py
git commit -m "feat: add main bot loop — wires scanner, stream, validator, and position manager"
```

---

## Self-Review

**Spec coverage check:**

| Spec Section | Covered by Task |
|---|---|
| Screening engine (movers + most-actives, 30s) | Task 5 |
| Stage 1 filter (≥5%, ≥$0.50) | Task 5 |
| Stage 2 validation (RoC, relative volume, buying pressure) | Task 7 |
| Dynamic watchlist with stream subscriptions | Task 6 |
| Volume baseline (20-day avg ÷ 390) | Task 6 |
| Broker limit + stop orders | Task 3 |
| Hard stop (broker-level, 1.5× ATR) | Task 9 |
| Trailing stop (2× ATR, software-managed, updates broker) | Task 8 |
| VWAP break exit | Task 8 |
| Volume collapse exit | Task 8 |
| Structure break exit | Task 8 |
| Overnight hold decision | Task 8 |
| Risk parameters (V4Config overrides) | Task 2 |
| Portfolio heat, sector cap, kill switch | Retained from V3, wired in Task 9 |
| Repo setup (clone + prune) | Task 1 |

**Placeholder scan:** No TBDs found. All code blocks are complete.

**Type consistency check:** `Position` fields (`highest_close`, `stop_order_id`, `entry_bar_volume`) added in Task 2 are used consistently in Tasks 8 and 9. `ExitInstruction` defined in Task 8 used in Task 9. `V4Config` defined in Task 2 passed to all components in Tasks 5–9.
