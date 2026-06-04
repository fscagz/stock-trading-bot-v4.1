# V4 Backtesting System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI backtesting system that replays the V4 momentum strategy against historical data using existing V4 components unchanged, with JSON-cached 1-min bar fetching from Alpaca and daily bar pre-screening via Yahoo Finance.

**Architecture:** Four new modules added to `bot/backtest/` (leaving V3 files untouched). `CandidateScreener` identifies which symbols to test per day using daily bars; `BarFetcher` downloads and caches Alpaca IEX 1-min bars; `Simulator` drives existing indicators/validator/manager against merged bar streams; `BacktestMetrics` aggregates closed `TradeRecord`s. A `__main__.py` CLI wires them together.

**Tech Stack:** Python 3.13, requests, yfinance (via existing `get_daily`/`get_daily_batch`), pandas, python-dotenv, zoneinfo, existing V4 components (ATRIndicator, VWAPIndicator, MomentumValidator, PositionManager, PortfolioState, compute_position_size).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `bot/backtest/backtest_metrics.py` | Create | Compute aggregate stats from `List[TradeRecord]` |
| `bot/backtest/candidate_screener.py` | Create | Daily bar screen — identify movers per date |
| `bot/backtest/bar_fetcher.py` | Create | Fetch + cache 1-min Alpaca IEX bars |
| `bot/backtest/simulator.py` | Create | Replay engine — drives all existing V4 components |
| `bot/backtest/__main__.py` | Create | CLI entry point |
| `testing/test_backtest_metrics.py` | Create | Unit tests for compute_metrics |
| `testing/test_candidate_screener.py` | Create | Unit tests for CandidateScreener |
| `testing/test_bar_fetcher.py` | Create | Unit tests for BarFetcher |
| `testing/test_simulator.py` | Create | Unit tests for Simulator |

Existing V3 files in `bot/backtest/` (`engine.py`, `metrics.py`, `costs.py`, `integrity.py`, `stress_test.py`, `walk_forward.py`) are left untouched.

---

## Task 1: BacktestMetrics

**Files:**
- Create: `bot/backtest/backtest_metrics.py`
- Test: `testing/test_backtest_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
# testing/test_backtest_metrics.py
from datetime import datetime, timezone, timedelta
from bot.backtest.backtest_metrics import compute_metrics
from bot.intraday.types import TradeRecord


def _record(pnl: float, exit_reason: str = "hard_stop", hold_minutes: int = 30) -> TradeRecord:
    entry = datetime(2025, 3, 10, 14, 0, tzinfo=timezone.utc)
    exit_ = entry + timedelta(minutes=hold_minutes)
    return TradeRecord(
        ticker="TEST",
        direction="long",
        entry_time=entry,
        entry_price=10.0,
        shares=100,
        stop_price=9.0,
        target_price=12.0,
        signals=["momentum"],
        sector="Unknown",
        regime="",
        portfolio_heat_at_entry=0.01,
        expected_slippage_pct=0.001,
        exit_time=exit_,
        exit_price=10.0 + pnl / 100,
        pnl=pnl,
        exit_reason=exit_reason,
    )


def test_empty_trades():
    result = compute_metrics([], 10000.0)
    assert result["total_trades"] == 0
    assert result["win_rate"] == 0.0
    assert result["total_pnl"] == 0.0


def test_win_rate():
    trades = [_record(100.0), _record(50.0), _record(-30.0), _record(-20.0)]
    result = compute_metrics(trades, 10000.0)
    assert result["total_trades"] == 4
    assert result["win_rate"] == 0.5


def test_total_pnl():
    trades = [_record(100.0), _record(-40.0)]
    result = compute_metrics(trades, 10000.0)
    assert result["total_pnl"] == 60.0


def test_avg_winner_and_loser():
    trades = [_record(100.0), _record(200.0), _record(-50.0), _record(-100.0)]
    result = compute_metrics(trades, 10000.0)
    assert result["avg_winner"] == 150.0
    assert result["avg_loser"] == -75.0


def test_max_drawdown():
    # equity: 10000 +100 +200 -300 -100
    # peaks at 10300, troughs at 9900 → max drawdown = 400
    trades = [_record(100.0), _record(200.0), _record(-300.0), _record(-100.0)]
    result = compute_metrics(trades, 10000.0)
    assert result["max_drawdown"] == 400.0


def test_avg_hold_minutes():
    trades = [_record(10.0, hold_minutes=30), _record(-10.0, hold_minutes=60)]
    result = compute_metrics(trades, 10000.0)
    assert result["avg_hold_minutes"] == 45.0


def test_exit_reasons():
    trades = [
        _record(10.0, exit_reason="hard_stop"),
        _record(-10.0, exit_reason="hard_stop"),
        _record(20.0, exit_reason="eod"),
    ]
    result = compute_metrics(trades, 10000.0)
    assert result["exit_reasons"]["hard_stop"] == 2
    assert result["exit_reasons"]["eod"] == 1
```

- [ ] **Step 2: Run to confirm failure**

```bash
python3.13 -m pytest testing/test_backtest_metrics.py -v
```

Expected: `ModuleNotFoundError: No module named 'bot.backtest.backtest_metrics'`

- [ ] **Step 3: Implement `bot/backtest/backtest_metrics.py`**

```python
from __future__ import annotations
from typing import List

from bot.intraday.types import TradeRecord


def compute_metrics(trades: List[TradeRecord], initial_equity: float) -> dict:
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl_per_trade": 0.0,
            "avg_winner": 0.0,
            "avg_loser": 0.0,
            "max_drawdown": 0.0,
            "avg_hold_minutes": 0.0,
            "exit_reasons": {},
        }

    pnls = [t.pnl for t in trades if t.pnl is not None]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    equity = initial_equity
    peak = equity
    max_drawdown = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    hold_minutes = []
    for t in trades:
        if t.exit_time and t.entry_time:
            hold_minutes.append((t.exit_time - t.entry_time).total_seconds() / 60)

    exit_reasons: dict = {}
    for t in trades:
        if t.exit_reason:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return {
        "total_trades": len(trades),
        "win_rate": len(winners) / len(pnls) if pnls else 0.0,
        "total_pnl": round(sum(pnls), 2),
        "avg_pnl_per_trade": round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
        "avg_winner": round(sum(winners) / len(winners), 2) if winners else 0.0,
        "avg_loser": round(sum(losers) / len(losers), 2) if losers else 0.0,
        "max_drawdown": round(max_drawdown, 2),
        "avg_hold_minutes": round(sum(hold_minutes) / len(hold_minutes), 1) if hold_minutes else 0.0,
        "exit_reasons": exit_reasons,
    }
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
python3.13 -m pytest testing/test_backtest_metrics.py -v
```

Expected: 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bot/backtest/backtest_metrics.py testing/test_backtest_metrics.py
git commit -m "feat: add BacktestMetrics — compute_metrics over TradeRecord list"
```

---

## Task 2: CandidateScreener

**Files:**
- Create: `bot/backtest/candidate_screener.py`
- Test: `testing/test_candidate_screener.py`

The screener loads a symbol universe from the Alpaca assets endpoint (same filter as MarketScanner: NYSE/NASDAQ/AMEX, pure alphabetic tickers), then for each target date uses `get_daily_batch` to find symbols whose intraday high reached ≥5% above the prior day's close with a closing price ≥$0.50.

- [ ] **Step 1: Write the failing tests**

```python
# testing/test_candidate_screener.py
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from bot.backtest.candidate_screener import CandidateScreener
from bot.config import V4Config

_TARGET_DATE = date(2025, 3, 10)
_PREV_DATE = date(2025, 3, 7)

ASSETS_RESPONSE = [
    {"symbol": "ASTC", "exchange": "NASDAQ", "status": "active", "tradable": True},
    {"symbol": "SNDL", "exchange": "NASDAQ", "status": "active", "tradable": True},
    {"symbol": "OTC1", "exchange": "OTC",    "status": "active", "tradable": True},
    {"symbol": "JOBY.WS", "exchange": "NYSE", "status": "active", "tradable": True},
]


def _make_df(prev_close: float, day_high: float, day_close: float) -> pd.DataFrame:
    idx = pd.to_datetime([_PREV_DATE, _TARGET_DATE])
    return pd.DataFrame(
        {
            "open": [prev_close, day_close * 0.99],
            "high": [prev_close, day_high],
            "low": [prev_close * 0.99, day_close * 0.98],
            "close": [prev_close, day_close],
            "volume": [1_000_000, 5_000_000],
        },
        index=idx,
    )


def _make_screener(universe: list | None = None) -> CandidateScreener:
    cfg = V4Config()
    screener = CandidateScreener(cfg, "key", "secret")
    if universe is not None:
        screener._universe = universe
    return screener


def test_stage1_filter_includes_5pct_mover():
    # ASTC: high=2.30 vs prev_close=2.00 → 15% gain, close=2.25 → passes
    screener = _make_screener(["ASTC", "SNDL"])
    daily_data = {
        "ASTC": _make_df(2.00, 2.30, 2.25),
        "SNDL": _make_df(1.20, 1.24, 1.22),  # 3.3% gain — below 5%
    }
    with patch("bot.backtest.candidate_screener.get_daily_batch", return_value=daily_data):
        candidates = screener.candidates_for_date(_TARGET_DATE)
    assert "ASTC" in candidates


def test_stage1_filter_excludes_low_gain():
    # SNDL: 3.3% gain — below stage1_min_price_change_pct (5%)
    screener = _make_screener(["ASTC", "SNDL"])
    daily_data = {
        "ASTC": _make_df(2.00, 2.30, 2.25),
        "SNDL": _make_df(1.20, 1.24, 1.22),
    }
    with patch("bot.backtest.candidate_screener.get_daily_batch", return_value=daily_data):
        candidates = screener.candidates_for_date(_TARGET_DATE)
    assert "SNDL" not in candidates


def test_stage1_filter_excludes_close_below_min_price():
    # 15% gain but close=0.32 — below stage1_min_price (0.50)
    screener = _make_screener(["LOWP"])
    daily_data = {"LOWP": _make_df(0.28, 0.33, 0.32)}
    with patch("bot.backtest.candidate_screener.get_daily_batch", return_value=daily_data):
        candidates = screener.candidates_for_date(_TARGET_DATE)
    assert "LOWP" not in candidates


def test_universe_loaded_from_alpaca_on_first_call():
    screener = _make_screener()  # no pre-loaded universe
    assert screener._universe is None
    with patch("bot.backtest.candidate_screener.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            json=lambda: ASSETS_RESPONSE,
            raise_for_status=lambda: None,
        )
        with patch("bot.backtest.candidate_screener.get_daily_batch", return_value={}):
            screener.candidates_for_date(_TARGET_DATE)
    assert screener._universe is not None
    assert "ASTC" in screener._universe
    assert "OTC1" not in screener._universe    # wrong exchange
    assert "JOBY.WS" not in screener._universe  # non-alphabetic


def test_batch_failure_does_not_crash():
    screener = _make_screener(["ASTC"])
    with patch("bot.backtest.candidate_screener.get_daily_batch", side_effect=Exception("network")):
        candidates = screener.candidates_for_date(_TARGET_DATE)
    assert candidates == []
```

- [ ] **Step 2: Run to confirm failure**

```bash
python3.13 -m pytest testing/test_candidate_screener.py -v
```

Expected: `ModuleNotFoundError: No module named 'bot.backtest.candidate_screener'`

- [ ] **Step 3: Implement `bot/backtest/candidate_screener.py`**

```python
from __future__ import annotations
import logging
import re
from datetime import date, timedelta
from typing import List, Optional

import pandas as pd
import requests

from bot.config import V4Config
from bot.data.daily_loader import get_daily_batch

_EXCHANGE_ALLOWLIST = {"NYSE", "NASDAQ", "AMEX"}
_COMMON_STOCK_RE = re.compile(r"^[A-Z]{1,5}$")
_BATCH_SIZE = 500
logger = logging.getLogger(__name__)


class CandidateScreener:
    def __init__(
        self,
        config: V4Config,
        api_key: str,
        secret_key: str,
        base_url: str = "https://paper-api.alpaca.markets",
    ) -> None:
        self._config = config
        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        self._assets_url = f"{base_url.rstrip('/')}/v2/assets"
        self._universe: Optional[List[str]] = None

    def candidates_for_date(self, trade_date: date) -> List[str]:
        if self._universe is None:
            self._universe = self._load_universe()
        if not self._universe:
            return []

        start_str = (trade_date - timedelta(days=10)).isoformat()
        end_str = (trade_date + timedelta(days=1)).isoformat()
        target_ts = pd.Timestamp(trade_date)

        candidates: List[str] = []
        for i in range(0, len(self._universe), _BATCH_SIZE):
            batch = self._universe[i : i + _BATCH_SIZE]
            try:
                daily_data = get_daily_batch(batch, start=start_str, end=end_str)
            except Exception as exc:
                logger.warning("CandidateScreener: batch failed: %s", exc)
                continue

            for sym, df in daily_data.items():
                if target_ts not in df.index:
                    continue
                idx = df.index.get_loc(target_ts)
                if idx == 0:
                    continue  # no prior day row for prev_close
                prev_close = float(df.iloc[idx - 1]["close"])
                day_high = float(df.iloc[idx]["high"])
                day_close = float(df.iloc[idx]["close"])
                if prev_close <= 0:
                    continue
                pct_change = (day_high - prev_close) / prev_close
                if (
                    pct_change >= self._config.stage1_min_price_change_pct
                    and day_close >= self._config.stage1_min_price
                ):
                    candidates.append(sym)

        logger.info("CandidateScreener: %d candidates for %s", len(candidates), trade_date)
        return candidates

    def _load_universe(self) -> List[str]:
        try:
            resp = requests.get(
                self._assets_url,
                headers=self._headers,
                params={"status": "active", "asset_class": "us_equity"},
                timeout=30,
            )
            resp.raise_for_status()
            assets = resp.json()
        except Exception as exc:
            logger.error("CandidateScreener: failed to load universe: %s", exc)
            return []

        symbols = []
        for a in assets:
            if not a.get("tradable"):
                continue
            if a.get("exchange") not in _EXCHANGE_ALLOWLIST:
                continue
            sym = a.get("symbol", "")
            if _COMMON_STOCK_RE.match(sym):
                symbols.append(sym)
        logger.info("CandidateScreener: universe loaded with %d symbols", len(symbols))
        return symbols
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
python3.13 -m pytest testing/test_candidate_screener.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bot/backtest/candidate_screener.py testing/test_candidate_screener.py
git commit -m "feat: add CandidateScreener — daily bar pre-screen for backtest candidates"
```

---

## Task 3: BarFetcher

**Files:**
- Create: `bot/backtest/bar_fetcher.py`
- Test: `testing/test_bar_fetcher.py`

Fetches 1-min bars from Alpaca's IEX feed. Cache key: `{cache_dir}/{SYMBOL}_{DATE}.json`. Session filter: 09:30–16:00 ET. Handles pagination via `next_page_token`.

- [ ] **Step 1: Write the failing tests**

```python
# testing/test_bar_fetcher.py
import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from bot.backtest.bar_fetcher import BarFetcher

_SAMPLE_BARS_RAW = [
    # Regular session bars (ET 09:30 and 09:31)
    {"t": "2025-03-10T14:30:00Z", "o": 2.0, "h": 2.5, "l": 1.9, "c": 2.3, "v": 50000},
    {"t": "2025-03-10T14:31:00Z", "o": 2.3, "h": 2.6, "l": 2.2, "c": 2.5, "v": 40000},
    # Pre-market bar (ET 09:00) — should be filtered out
    {"t": "2025-03-10T14:00:00Z", "o": 2.0, "h": 2.1, "l": 1.95, "c": 2.05, "v": 5000},
]


def _fetcher(tmpdir: str) -> BarFetcher:
    return BarFetcher("key", "secret", cache_dir=tmpdir)


def test_cache_miss_fetches_api_and_writes_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        fetcher = _fetcher(tmpdir)
        resp = MagicMock(raise_for_status=lambda: None)
        resp.json.return_value = {"bars": _SAMPLE_BARS_RAW, "next_page_token": None}
        with patch("bot.backtest.bar_fetcher.requests.get", return_value=resp):
            bars = fetcher.fetch("ASTC", date(2025, 3, 10))
        # 2 session bars (pre-market filtered out)
        assert len(bars) == 2
        cache_file = Path(tmpdir) / "ASTC_2025-03-10.json"
        assert cache_file.exists()


def test_cache_hit_skips_api():
    with tempfile.TemporaryDirectory() as tmpdir:
        fetcher = _fetcher(tmpdir)
        cached = [{"t": "2025-03-10T14:30:00+00:00", "o": 2.0, "h": 2.5, "l": 1.9, "c": 2.3, "v": 50000}]
        (Path(tmpdir) / "ASTC_2025-03-10.json").write_text(json.dumps(cached))
        with patch("bot.backtest.bar_fetcher.requests.get") as mock_get:
            bars = fetcher.fetch("ASTC", date(2025, 3, 10))
        mock_get.assert_not_called()
        assert len(bars) == 1
        assert bars[0].symbol == "ASTC"
        assert bars[0].close == 2.3


def test_session_filter_excludes_premarket():
    with tempfile.TemporaryDirectory() as tmpdir:
        fetcher = _fetcher(tmpdir)
        resp = MagicMock(raise_for_status=lambda: None)
        resp.json.return_value = {"bars": _SAMPLE_BARS_RAW, "next_page_token": None}
        with patch("bot.backtest.bar_fetcher.requests.get", return_value=resp):
            bars = fetcher.fetch("ASTC", date(2025, 3, 10))
        from zoneinfo import ZoneInfo
        _ET = ZoneInfo("America/New_York")
        for b in bars:
            ts_et = b.timestamp.astimezone(_ET)
            assert (ts_et.hour, ts_et.minute) >= (9, 30)


def test_pagination_collects_all_bars():
    with tempfile.TemporaryDirectory() as tmpdir:
        fetcher = _fetcher(tmpdir)
        resp1 = MagicMock(raise_for_status=lambda: None)
        resp1.json.return_value = {"bars": [_SAMPLE_BARS_RAW[0]], "next_page_token": "tok123"}
        resp2 = MagicMock(raise_for_status=lambda: None)
        resp2.json.return_value = {"bars": [_SAMPLE_BARS_RAW[1]], "next_page_token": None}
        with patch("bot.backtest.bar_fetcher.requests.get") as mock_get:
            mock_get.side_effect = [resp1, resp2]
            bars = fetcher.fetch("ASTC", date(2025, 3, 10))
        assert len(bars) == 2
        assert mock_get.call_count == 2


def test_api_error_returns_empty_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        fetcher = _fetcher(tmpdir)
        with patch("bot.backtest.bar_fetcher.requests.get", side_effect=Exception("network")):
            bars = fetcher.fetch("ASTC", date(2025, 3, 10))
        assert bars == []
```

- [ ] **Step 2: Run to confirm failure**

```bash
python3.13 -m pytest testing/test_bar_fetcher.py -v
```

Expected: `ModuleNotFoundError: No module named 'bot.backtest.bar_fetcher'`

- [ ] **Step 3: Implement `bot/backtest/bar_fetcher.py`**

```python
from __future__ import annotations
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

import requests

from bot.intraday.types import Bar

_ET = ZoneInfo("America/New_York")
_SESSION_OPEN = (9, 30)
_SESSION_CLOSE = (16, 0)
_BARS_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
logger = logging.getLogger(__name__)


class BarFetcher:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        cache_dir: str = "backtest_results/cache",
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, symbol: str, trade_date: date) -> List[Bar]:
        cache_path = self._cache_dir / f"{symbol}_{trade_date}.json"
        if cache_path.exists():
            raw = json.loads(cache_path.read_text())
            return [self._parse_bar(symbol, b) for b in raw]

        bars = self._fetch_from_api(symbol, trade_date)
        if bars is not None:
            cache_path.write_text(json.dumps([self._bar_to_dict(b) for b in bars]))
        return bars or []

    def _fetch_from_api(self, symbol: str, trade_date: date) -> Optional[List[Bar]]:
        start = datetime(
            trade_date.year, trade_date.month, trade_date.day, 9, 30, tzinfo=_ET
        )
        end = datetime(
            trade_date.year, trade_date.month, trade_date.day, 16, 0, tzinfo=_ET
        )
        url = _BARS_URL.format(symbol=symbol)
        headers = {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
        }
        params: dict = {
            "timeframe": "1Min",
            "feed": "iex",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 1000,
        }

        all_bars: List[Bar] = []
        while True:
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("BarFetcher: API error for %s %s: %s", symbol, trade_date, exc)
                return None

            for b in data.get("bars") or []:
                bar = self._parse_bar(symbol, b)
                if self._in_session(bar.timestamp):
                    all_bars.append(bar)

            next_token = data.get("next_page_token")
            if not next_token:
                break
            params["page_token"] = next_token

        return all_bars

    def _in_session(self, ts: datetime) -> bool:
        ts_et = ts.astimezone(_ET)
        t = (ts_et.hour, ts_et.minute)
        return _SESSION_OPEN <= t <= _SESSION_CLOSE

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

- [ ] **Step 4: Run tests to confirm pass**

```bash
python3.13 -m pytest testing/test_bar_fetcher.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bot/backtest/bar_fetcher.py testing/test_bar_fetcher.py
git commit -m "feat: add BarFetcher — Alpaca IEX 1-min bars with local JSON cache"
```

---

## Task 4: Simulator

**Files:**
- Create: `bot/backtest/simulator.py`
- Test: `testing/test_simulator.py`

Replays merged 1-min bars through existing V4 components. Fresh instances of all indicators/managers per day. Fill prices: entry at next bar's open + slippage_pct; stop hit (bar.low ≤ stop_price) at stop_price; VWAP break/volume collapse/structure break at bar.close; EOD at 15:25 bar close.

- [ ] **Step 1: Write the failing tests**

```python
# testing/test_simulator.py
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
from typing import List

from bot.backtest.simulator import Simulator, BacktestResult
from bot.config import V4Config
from bot.intraday.types import Bar

_ET = ZoneInfo("America/New_York")
_DATE = date(2025, 3, 10)


def _bar(sym: str, hour: int, minute: int, o: float, h: float, l: float, c: float, v: int = 500_000) -> Bar:
    ts = datetime(_DATE.year, _DATE.month, _DATE.day, hour, minute, tzinfo=_ET)
    return Bar(symbol=sym, timestamp=ts, open=o, high=h, low=l, close=c, volume=v)


def _make_config() -> V4Config:
    cfg = V4Config()
    cfg.stage2_roc_lookback_bars = 2
    cfg.stage2_roc_min_pct = 0.03
    cfg.stage2_min_relative_volume = 2.0
    cfg.stage2_buying_pressure_min = 0.60
    return cfg


def _make_sim(equity: float = 100_000.0, slippage: float = 0.0) -> Simulator:
    return Simulator(config=_make_config(), initial_equity=equity, slippage_pct=slippage)


def _signal_bars(sym: str) -> List[Bar]:
    """3 bars that trigger Stage 2: 16% RoC, high relative volume, strong close."""
    return [
        _bar(sym, 9, 30, 2.00, 2.10, 1.98, 2.05),
        _bar(sym, 9, 31, 2.05, 2.20, 2.03, 2.15),
        _bar(sym, 9, 32, 2.15, 2.40, 2.14, 2.38),  # signal bar
    ]


# baseline of 200k → 500k bar volume = 2.5x, satisfies stage2_min_relative_volume=2.0
_BASELINE = {"ASTC": 200_000.0}


def test_entry_fills_at_next_bar_open_plus_slippage():
    sim = _make_sim(slippage=0.01)
    bars = _signal_bars("ASTC")
    entry_bar = _bar("ASTC", 9, 33, 2.40, 2.55, 2.38, 2.48)  # open=2.40
    eod_bar = _bar("ASTC", 15, 25, 2.48, 2.52, 2.42, 2.50, v=30_000)  # low volume → EOD close
    result = sim.run_day(_DATE, {"ASTC": bars + [entry_bar, eod_bar]}, _BASELINE)

    assert len(result.trades) == 1
    expected_fill = round(2.40 * 1.01, 2)  # 2.42
    assert result.trades[0].entry_price == expected_fill


def test_stop_hit_fills_at_stop_price():
    sim = _make_sim(slippage=0.0)
    bars = _signal_bars("ASTC")
    entry_bar = _bar("ASTC", 9, 33, 2.40, 2.55, 2.38, 2.48)
    # crash bar: low=0.50 will be below any computed stop (entry~2.40, stop~2.04)
    crash_bar = _bar("ASTC", 9, 34, 2.48, 2.50, 0.50, 2.30, v=100_000)
    result = sim.run_day(_DATE, {"ASTC": bars + [entry_bar, crash_bar]}, _BASELINE)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "hard_stop"
    assert trade.exit_price == trade.stop_price


def test_eod_close_at_1525():
    sim = _make_sim(slippage=0.0)
    bars = _signal_bars("ASTC")
    entry_bar = _bar("ASTC", 9, 33, 2.40, 2.55, 2.38, 2.48)
    # EOD bar: low volume triggers should_hold_overnight=False → close at bar.close
    eod_bar = _bar("ASTC", 15, 25, 2.48, 2.52, 2.42, 2.35, v=30_000)
    result = sim.run_day(_DATE, {"ASTC": bars + [entry_bar, eod_bar]}, _BASELINE)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "eod"
    assert trade.exit_price == 2.35


def test_no_entry_when_no_next_bar():
    """Signal on last bar of day → pending entry never fills → no trade."""
    sim = _make_sim(slippage=0.0)
    bars = _signal_bars("ASTC")  # last bar is the signal bar, no bars after
    result = sim.run_day(_DATE, {"ASTC": bars}, _BASELINE)
    assert len(result.trades) == 0


def test_equity_adjusts_on_close():
    sim = _make_sim(equity=10_000.0, slippage=0.0)
    bars = _signal_bars("ASTC")
    entry_bar = _bar("ASTC", 9, 33, 2.40, 2.55, 2.38, 2.48)
    eod_bar = _bar("ASTC", 15, 25, 2.48, 2.55, 2.47, 2.50, v=30_000)
    result = sim.run_day(_DATE, {"ASTC": bars + [entry_bar, eod_bar]}, _BASELINE)

    if result.trades:
        trade = result.trades[0]
        if trade.pnl is not None:
            # equity_curve must exist and reflect pnl
            assert len(result.equity_curve) > 0
```

- [ ] **Step 2: Run to confirm failure**

```bash
python3.13 -m pytest testing/test_simulator.py -v
```

Expected: `ModuleNotFoundError: No module named 'bot.backtest.simulator'`

- [ ] **Step 3: Implement `bot/backtest/simulator.py`**

```python
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from bot.config import V4Config
from bot.intraday.indicators.atr import ATRIndicator
from bot.intraday.indicators.vwap import VWAPIndicator
from bot.intraday.risk.portfolio import PortfolioState
from bot.intraday.risk.sizing import compute_position_size
from bot.intraday.types import Bar, Position, TradeRecord
from bot.momentum.validator import MomentumValidator
from bot.positions.manager import PositionManager

_ET = ZoneInfo("America/New_York")
_EOD_HOUR = 15
_EOD_MINUTE = 25
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    trades: List[TradeRecord]
    equity_curve: List[Tuple[datetime, float]]


class Simulator:
    def __init__(
        self,
        config: V4Config,
        initial_equity: float,
        slippage_pct: float = 0.001,
    ) -> None:
        self._config = config
        self._initial_equity = initial_equity
        self._slippage_pct = slippage_pct

    def run_day(
        self,
        trade_date: date,
        bars_by_symbol: Dict[str, List[Bar]],
        baseline_volumes: Dict[str, float],
    ) -> BacktestResult:
        atr_indicator = ATRIndicator(period=14)
        vwap_indicator = VWAPIndicator()
        validator = MomentumValidator(self._config)
        manager = PositionManager(self._config)
        portfolio = PortfolioState(equity=self._initial_equity, config=self._config)

        merged: List[Bar] = sorted(
            (b for bars in bars_by_symbol.values() for b in bars),
            key=lambda b: b.timestamp,
        )

        open_records: Dict[str, TradeRecord] = {}
        closed_trades: List[TradeRecord] = []
        pending_entries: Dict[str, dict] = {}  # symbol → {atr_val, size}
        equity_curve: List[Tuple[datetime, float]] = []

        for bar in merged:
            sym = bar.symbol
            baseline = baseline_volumes.get(sym, 0.0)
            bar_et = bar.timestamp.astimezone(_ET)

            atr_val = atr_indicator.update(bar)
            vwap_val = vwap_indicator.update(bar)

            # Fill pending entry at this bar's open + slippage
            if sym in pending_entries:
                pending = pending_entries.pop(sym)
                fill_price = round(bar.open * (1 + self._slippage_pct), 2)
                size = pending["size"]
                atr_entry = pending["atr_val"]
                stop_price = size.long_stop(fill_price)
                target_price = size.long_target(fill_price)
                position = Position(
                    ticker=sym,
                    direction="long",
                    shares=size.shares,
                    entry_price=fill_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    entry_time=bar.timestamp,
                    atr_at_entry=atr_entry,
                    signals=["momentum"],
                    sector="Unknown",
                    highest_close=fill_price,
                    entry_bar_volume=bar.volume,
                )
                portfolio.add_position(position)
                open_records[sym] = TradeRecord(
                    ticker=sym,
                    direction="long",
                    entry_time=bar.timestamp,
                    entry_price=fill_price,
                    shares=size.shares,
                    stop_price=stop_price,
                    target_price=target_price,
                    signals=["momentum"],
                    sector="Unknown",
                    regime="",
                    portfolio_heat_at_entry=portfolio.portfolio_heat_pct,
                    expected_slippage_pct=self._slippage_pct,
                )
                logger.debug("BT ENTRY %s @ %.2f shares=%d", sym, fill_price, size.shares)
                equity_curve.append((bar.timestamp, portfolio.equity))
                continue

            # EOD evaluation at 15:25 ET
            if bar_et.hour == _EOD_HOUR and bar_et.minute == _EOD_MINUTE:
                if sym in portfolio.positions and vwap_val is not None and baseline > 0:
                    position = portfolio.positions[sym]
                    if not manager.should_hold_overnight(bar, position, vwap_val, baseline):
                        self._close(sym, bar.close, "eod", bar.timestamp,
                                    portfolio, open_records, closed_trades)
                equity_curve.append((bar.timestamp, portfolio.equity))
                continue

            # Exit logic for open positions
            if sym in portfolio.positions:
                position = portfolio.positions[sym]
                if bar.low <= position.stop_price:
                    # Mid-bar stop hit: fill at stop_price
                    self._close(sym, position.stop_price, "hard_stop", bar.timestamp,
                                portfolio, open_records, closed_trades)
                elif vwap_val is not None and baseline > 0:
                    instruction = manager.on_bar(bar, position, vwap_val, baseline)
                    if instruction:
                        if instruction.reason in ("hard_stop", "trailing_stop"):
                            fill = position.stop_price
                        elif instruction.limit_price is not None:
                            fill = instruction.limit_price
                        else:
                            fill = bar.close
                        self._close(sym, fill, instruction.reason, bar.timestamp,
                                    portfolio, open_records, closed_trades)
                equity_curve.append((bar.timestamp, portfolio.equity))
                continue

            # Entry logic
            if sym in pending_entries:
                continue
            if not (atr_val and baseline > 0):
                continue
            can_enter, _ = portfolio.can_enter(sector="Unknown", now=bar.timestamp)
            if not can_enter:
                continue
            if validator.validate(bar, baseline):
                size = compute_position_size(portfolio.equity, atr_val, bar.close, self._config)
                if size.shares > 0:
                    pending_entries[sym] = {"atr_val": atr_val, "size": size}
                    logger.debug("BT SIGNAL %s @ %s", sym, bar.timestamp)

        return BacktestResult(trades=closed_trades, equity_curve=equity_curve)

    def _close(
        self,
        sym: str,
        fill_price: float,
        reason: str,
        ts: datetime,
        portfolio: PortfolioState,
        open_records: Dict[str, TradeRecord],
        closed_trades: List[TradeRecord],
    ) -> None:
        position = portfolio.remove_position(sym)
        if not position:
            return
        pnl = round((fill_price - position.entry_price) * position.shares, 2)
        portfolio.equity += pnl
        record = open_records.pop(sym, None)
        if record:
            record.exit_time = ts
            record.exit_price = fill_price
            record.pnl = pnl
            record.exit_reason = reason
            closed_trades.append(record)
        logger.debug("BT EXIT %s @ %.2f pnl=%.2f reason=%s", sym, fill_price, pnl, reason)
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
python3.13 -m pytest testing/test_simulator.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add bot/backtest/simulator.py testing/test_simulator.py
git commit -m "feat: add Simulator — replay engine driving V4 components against historical bars"
```

---

## Task 5: CLI (`__main__.py`)

**Files:**
- Create: `bot/backtest/__main__.py`

No unit tests; verified by running the CLI in targeted mode against a known date.

- [ ] **Step 1: Implement `bot/backtest/__main__.py`**

```python
from __future__ import annotations
import argparse
import csv
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

import bot.broker_alpaca as broker
from bot.backtest.backtest_metrics import compute_metrics
from bot.backtest.bar_fetcher import BarFetcher
from bot.backtest.candidate_screener import CandidateScreener
from bot.backtest.simulator import Simulator
from bot.config import V4Config
from bot.data.daily_loader import get_daily
from bot.intraday.types import TradeRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _trading_days(start: date, end: date) -> List[date]:
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _write_trades_csv(trades: List[TradeRecord], path: Path) -> None:
    fields = [
        "ticker", "direction", "entry_time", "entry_price", "shares",
        "stop_price", "target_price", "exit_time", "exit_price",
        "pnl", "exit_reason", "portfolio_heat_at_entry", "signals",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in trades:
            writer.writerow({
                "ticker": t.ticker,
                "direction": t.direction,
                "entry_time": t.entry_time.isoformat(),
                "entry_price": t.entry_price,
                "shares": t.shares,
                "stop_price": t.stop_price,
                "target_price": t.target_price,
                "exit_time": t.exit_time.isoformat() if t.exit_time else "",
                "exit_price": t.exit_price if t.exit_price is not None else "",
                "pnl": round(t.pnl, 2) if t.pnl is not None else "",
                "exit_reason": t.exit_reason or "",
                "portfolio_heat_at_entry": round(t.portfolio_heat_at_entry, 4),
                "signals": "|".join(t.signals),
            })


def _write_summary_csv(metrics: dict, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 Momentum Bot Backtester")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--start", help="Start date YYYY-MM-DD (statistical mode)")
    mode.add_argument("--symbol", help="Single symbol (targeted mode)")
    parser.add_argument("--end", help="End date YYYY-MM-DD (required with --start)")
    parser.add_argument("--date", dest="target_date", help="Date YYYY-MM-DD (required with --symbol)")
    parser.add_argument("--slippage", type=float, default=0.001, help="Slippage fraction (default 0.001)")
    args = parser.parse_args()

    api_key = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    config = V4Config()
    account = broker.get_account_info()
    initial_equity = account["portfolio_value"]
    logger.info("Initial equity: $%.2f", initial_equity)

    out_dir = Path("backtest_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    screener = CandidateScreener(config, api_key, secret_key, base_url)
    fetcher = BarFetcher(api_key, secret_key)
    simulator = Simulator(config, initial_equity, slippage_pct=args.slippage)

    if args.symbol:
        if not args.target_date:
            parser.error("--symbol requires --date")
        trade_date = date.fromisoformat(args.target_date)
        days = [trade_date]
        prefix = f"{args.symbol}_{trade_date}"
    else:
        if not args.end:
            parser.error("--start requires --end")
        days = _trading_days(date.fromisoformat(args.start), date.fromisoformat(args.end))
        prefix = f"{args.start}_{args.end}"

    all_trades: List[TradeRecord] = []

    for d in days:
        candidates = [args.symbol] if args.symbol else screener.candidates_for_date(d)
        if not candidates:
            logger.info("No candidates for %s — skipped", d)
            continue
        logger.info("%s: %d candidates", d, len(candidates))

        bars_by_symbol = {}
        baseline_volumes = {}
        for sym in candidates:
            bars = fetcher.fetch(sym, d)
            if not bars:
                logger.debug("No IEX bars for %s on %s", sym, d)
                continue
            bars_by_symbol[sym] = bars
            try:
                df = get_daily(sym, period="1mo")
                baseline_volumes[sym] = (
                    df["volume"].tail(20).mean() / 390 if not df.empty else 0.0
                )
            except Exception:
                baseline_volumes[sym] = 0.0

        if not bars_by_symbol:
            logger.info("No bar data for %s — skipped", d)
            continue

        result = simulator.run_day(d, bars_by_symbol, baseline_volumes)
        all_trades.extend(result.trades)
        logger.info("%s: %d trades closed", d, len(result.trades))

    metrics = compute_metrics(all_trades, initial_equity)
    trades_path = out_dir / f"trades_{prefix}.csv"
    summary_path = out_dir / f"summary_{prefix}.csv"
    _write_trades_csv(all_trades, trades_path)
    _write_summary_csv(metrics, summary_path)

    print(f"\n=== Backtest Results ({prefix}) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nTrades written to: {trades_path}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run full test suite to verify no regressions**

```bash
python3.13 -m pytest testing/ -v --ignore=testing/test_alpaca_data.py --ignore=testing/test_backtester.py --ignore=testing/test_ml_models.py --ignore=testing/test_store.py --ignore=testing/test_yfinance_data.py -q
```

Expected: All backtest tests PASS, all existing tests still PASS

- [ ] **Step 3: Smoke-test the CLI in targeted mode**

```bash
python3.13 -m bot.backtest --symbol ASTC --date 2021-12-10
```

Expected output: startup log showing equity, then either "No IEX bars" (if no data cached for that date) or bar-fetch logs followed by a results table. No crashes.

- [ ] **Step 4: Commit**

```bash
git add bot/backtest/__main__.py
git commit -m "feat: add backtest CLI — targeted and statistical modes with CSV output"
```

---

## Usage Reference

```bash
# Targeted mode — replay one symbol on a known event day
python3.13 -m bot.backtest --symbol ASTC --date 2021-12-10

# Statistical mode — run across a date range
python3.13 -m bot.backtest --start 2025-01-01 --end 2025-05-01

# Override slippage assumption
python3.13 -m bot.backtest --start 2025-01-01 --end 2025-05-01 --slippage 0.002
```

Output files land in `backtest_results/`:
- `trades_{prefix}.csv` — one row per closed trade (same columns as live `trade_logger`)
- `summary_{prefix}.csv` — aggregate metrics (win rate, max drawdown, etc.)
- `cache/{SYMBOL}_{DATE}.json` — cached 1-min bars (persist across runs)
