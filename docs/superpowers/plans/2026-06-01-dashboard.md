# Trading Bot Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a FastAPI web dashboard that runs in a background thread alongside `python3 -m bot.main`, displaying live account metrics, open positions, trade history, P&L chart, risk metrics, and config via a Slate Dark + Cyan themed single-page app.

**Architecture:** A `DashboardState` object is mutated by `on_bar` as the bot runs and read by a FastAPI server's `/api/state` endpoint. The browser polls `/api/state` every 3 seconds and updates the DOM. No Alpaca API calls are made by the dashboard.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, Chart.js (CDN), vanilla JS

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `requirements.txt` | Modify | Add fastapi, uvicorn |
| `bot/dashboard/__init__.py` | Create | Package marker |
| `bot/dashboard/state.py` | Create | Thread-safe shared state dataclass |
| `bot/dashboard/server.py` | Create | FastAPI app + background thread launcher |
| `bot/dashboard/templates.py` | Create | Full dashboard HTML as a string constant |
| `bot/main.py` | Modify | Wire DashboardState into on_bar and startup |
| `testing/test_dashboard_state.py` | Create | Tests for DashboardState |
| `testing/test_dashboard_server.py` | Create | Tests for /api/state and / endpoints |

---

## Task 1: Add dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add fastapi and uvicorn to requirements.txt**

Open `requirements.txt` and add the two new lines so it reads:

```
# Core
alpaca-py
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

- [ ] **Step 2: Install the new dependencies**

```bash
pip install fastapi uvicorn
```

Expected: Packages install without errors.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add fastapi and uvicorn for dashboard"
```

---

## Task 2: DashboardState

**Files:**
- Create: `bot/dashboard/__init__.py`
- Create: `bot/dashboard/state.py`
- Create: `testing/test_dashboard_state.py`

- [ ] **Step 1: Write the failing tests**

Create `testing/test_dashboard_state.py`:

```python
from __future__ import annotations
import threading
from datetime import datetime, timezone

import pytest

from bot.dashboard.state import DashboardState
from bot.intraday.types import Position


def _make_position(ticker: str = "ASTC") -> Position:
    return Position(
        ticker=ticker,
        direction="long",
        shares=100,
        entry_price=4.00,
        stop_price=3.70,
        target_price=4.60,
        entry_time=datetime(2026, 6, 1, 9, 30, tzinfo=timezone.utc),
        atr_at_entry=0.30,
        signals=["momentum"],
        sector="Unknown",
    )


def test_initial_state():
    s = DashboardState()
    assert s.equity == 0.0
    assert s.cash == 0.0
    assert s.buying_power == 0.0
    assert s.is_paper is True
    assert s.positions == {}
    assert s.closed_trades == []
    assert s.last_prices == {}
    assert s.config_snapshot == {}
    assert s.portfolio_heat_pct == 0.0
    assert s.kill_switch_active is False
    assert s.consecutive_losses == 0
    assert s.cooldown_until is None
    assert s.regime_uptrend is True


def test_lock_is_rlock():
    s = DashboardState()
    assert isinstance(s._lock, type(threading.RLock()))


def test_thread_safe_reads_and_writes():
    s = DashboardState()
    errors: list = []

    def writer():
        for i in range(200):
            with s._lock:
                s.equity = float(i)
                s.last_prices["ASTC"] = float(i) * 0.1

    def reader():
        for _ in range(200):
            with s._lock:
                _ = s.equity
                _ = dict(s.last_prices)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_positions_dict_is_independent_after_assignment():
    s = DashboardState()
    pos = _make_position()
    original = {"ASTC": pos}
    s.positions = dict(original)
    original["HOOK"] = _make_position("HOOK")
    assert "HOOK" not in s.positions
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest testing/test_dashboard_state.py -v
```

Expected: `ModuleNotFoundError: No module named 'bot.dashboard'`

- [ ] **Step 3: Create the package init**

Create `bot/dashboard/__init__.py` (empty):

```python
```

- [ ] **Step 4: Create DashboardState**

Create `bot/dashboard/state.py`:

```python
from __future__ import annotations
import threading
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from bot.intraday.types import Position, TradeRecord


class DashboardState:
    def __init__(self) -> None:
        # Account snapshot (populated once at startup)
        self.equity: float = 0.0
        self.cash: float = 0.0
        self.buying_power: float = 0.0
        self.is_paper: bool = True

        # Portfolio (synced from PortfolioState on every position change)
        self.positions: Dict[str, Position] = {}
        self.portfolio_heat_pct: float = 0.0
        self.kill_switch_active: bool = False
        self.consecutive_losses: int = 0
        self.cooldown_until: Optional[datetime] = None

        # Regime (synced once per trading day)
        self.regime_uptrend: bool = True
        self.regime_date: Optional[date] = None

        # Trades closed this session
        self.closed_trades: List[TradeRecord] = []

        # Last bar close price per symbol (for unrealized P&L)
        self.last_prices: Dict[str, float] = {}

        # Config key-value snapshot (populated once at startup)
        self.config_snapshot: Dict[str, Any] = {}

        self._lock = threading.RLock()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest testing/test_dashboard_state.py -v
```

Expected: 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add bot/dashboard/__init__.py bot/dashboard/state.py testing/test_dashboard_state.py
git commit -m "feat: add DashboardState — thread-safe shared state for dashboard"
```

---

## Task 3: FastAPI server

**Files:**
- Create: `bot/dashboard/server.py`
- Create: `testing/test_dashboard_server.py`

- [ ] **Step 1: Write the failing tests**

Create `testing/test_dashboard_server.py`:

```python
from __future__ import annotations
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from bot.dashboard.server import create_app
from bot.dashboard.state import DashboardState
from bot.intraday.types import Position, TradeRecord


def _make_position(ticker: str = "ASTC", entry_price: float = 4.00) -> Position:
    return Position(
        ticker=ticker,
        direction="long",
        shares=100,
        entry_price=entry_price,
        stop_price=entry_price * 0.925,
        target_price=entry_price * 1.15,
        entry_time=datetime(2026, 6, 1, 9, 30, tzinfo=timezone.utc),
        atr_at_entry=0.30,
        signals=["momentum"],
        sector="Unknown",
    )


def _make_closed_trade(ticker: str = "MSOX", pnl: float = 56.40) -> TradeRecord:
    return TradeRecord(
        ticker=ticker,
        direction="long",
        entry_time=datetime(2026, 6, 1, 9, 35, tzinfo=timezone.utc),
        entry_price=5.14,
        shares=120,
        stop_price=4.75,
        target_price=5.90,
        signals=["momentum"],
        sector="Unknown",
        regime="uptrend",
        portfolio_heat_at_entry=0.05,
        expected_slippage_pct=0.001,
        exit_time=datetime(2026, 6, 1, 9, 42, tzinfo=timezone.utc),
        exit_price=5.61,
        pnl=pnl,
        exit_reason="target",
    )


# ── /api/state ──────────────────────────────────────────────────

def test_api_state_empty_returns_200():
    state = DashboardState()
    client = TestClient(create_app(state))
    r = client.get("/api/state")
    assert r.status_code == 200


def test_api_state_shape_when_empty():
    state = DashboardState()
    client = TestClient(create_app(state))
    data = client.get("/api/state").json()
    for key in ("equity", "cash", "buying_power", "is_paper", "day_pnl",
                "day_pnl_pct", "regime_uptrend", "kill_switch_active",
                "portfolio_heat_pct", "max_portfolio_heat", "consecutive_losses",
                "cooldown_until", "open_positions_count", "max_open_positions",
                "positions", "closed_trades", "config"):
        assert key in data, f"missing key: {key}"
    assert data["positions"] == []
    assert data["closed_trades"] == []
    assert data["day_pnl"] == 0.0


def test_api_state_unrealized_pnl_computed_from_last_price():
    state = DashboardState()
    state.equity = 10_000.0
    pos = _make_position("ASTC", entry_price=4.00)
    with state._lock:
        state.positions["ASTC"] = pos
        state.last_prices["ASTC"] = 4.50  # +$0.50 × 100 shares = +$50

    data = TestClient(create_app(state)).get("/api/state").json()
    assert len(data["positions"]) == 1
    p = data["positions"][0]
    assert p["ticker"] == "ASTC"
    assert p["last_price"] == 4.50
    assert p["unrealized_pnl"] == pytest.approx(50.0)


def test_api_state_unrealized_pnl_falls_back_to_entry_price():
    state = DashboardState()
    pos = _make_position("HOOK", entry_price=8.00)
    with state._lock:
        state.positions["HOOK"] = pos
        # no last_prices entry

    data = TestClient(create_app(state)).get("/api/state").json()
    assert data["positions"][0]["unrealized_pnl"] == pytest.approx(0.0)


def test_api_state_day_pnl_is_sum_of_realized_plus_unrealized():
    state = DashboardState()
    state.equity = 10_000.0
    trade = _make_closed_trade(pnl=56.40)
    pos = _make_position("ASTC", entry_price=4.00)
    with state._lock:
        state.closed_trades.append(trade)
        state.positions["ASTC"] = pos
        state.last_prices["ASTC"] = 4.50  # unrealized = +50.0

    data = TestClient(create_app(state)).get("/api/state").json()
    assert data["day_pnl"] == pytest.approx(56.40 + 50.0)


def test_api_state_positions_sorted_newest_first():
    state = DashboardState()
    pos1 = _make_position("ASTC")
    pos1.entry_time = datetime(2026, 6, 1, 9, 30, tzinfo=timezone.utc)
    pos2 = _make_position("HOOK")
    pos2.entry_time = datetime(2026, 6, 1, 9, 45, tzinfo=timezone.utc)
    with state._lock:
        state.positions = {"ASTC": pos1, "HOOK": pos2}

    data = TestClient(create_app(state)).get("/api/state").json()
    tickers = [p["ticker"] for p in data["positions"]]
    assert tickers == ["HOOK", "ASTC"]


def test_api_state_closed_trades_sorted_newest_first():
    state = DashboardState()
    t1 = _make_closed_trade("MSOX")
    t1.exit_time = datetime(2026, 6, 1, 9, 42, tzinfo=timezone.utc)
    t2 = _make_closed_trade("DRUG")
    t2.exit_time = datetime(2026, 6, 1, 10, 31, tzinfo=timezone.utc)
    with state._lock:
        state.closed_trades = [t1, t2]

    data = TestClient(create_app(state)).get("/api/state").json()
    tickers = [t["ticker"] for t in data["closed_trades"]]
    assert tickers == ["DRUG", "MSOX"]


# ── / (HTML) ────────────────────────────────────────────────────

def test_index_returns_html():
    state = DashboardState()
    r = TestClient(create_app(state)).get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<html" in r.text.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest testing/test_dashboard_server.py -v
```

Expected: `ModuleNotFoundError: No module named 'bot.dashboard.server'`

- [ ] **Step 3: Implement server.py**

Create `bot/dashboard/server.py`:

```python
from __future__ import annotations
import logging
import os
import threading
from datetime import timezone

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from bot.dashboard.state import DashboardState
from bot.dashboard.templates import DASHBOARD_HTML

logger = logging.getLogger(__name__)


def create_app(state: DashboardState) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return DASHBOARD_HTML

    @app.get("/api/state")
    async def api_state() -> dict:
        with state._lock:
            realized_pnl = sum(
                r.pnl for r in state.closed_trades if r.pnl is not None
            )
            unrealized_pnl = sum(
                (state.last_prices.get(ticker, pos.entry_price) - pos.entry_price)
                * pos.shares
                for ticker, pos in state.positions.items()
            )
            day_pnl = round(realized_pnl + unrealized_pnl, 2)
            day_pnl_pct = round(day_pnl / state.equity, 6) if state.equity > 0 else 0.0

            positions = []
            for ticker, pos in state.positions.items():
                last_price = state.last_prices.get(ticker, pos.entry_price)
                unrealized = round(
                    (last_price - pos.entry_price) * pos.shares, 2
                )
                positions.append({
                    "ticker": ticker,
                    "shares": pos.shares,
                    "entry_price": pos.entry_price,
                    "stop_price": pos.stop_price,
                    "target_price": pos.target_price,
                    "last_price": last_price,
                    "unrealized_pnl": unrealized,
                    "open_risk": round(pos.open_risk, 2),
                    "entry_time": pos.entry_time.astimezone(timezone.utc).isoformat(),
                })
            positions.sort(key=lambda p: p["entry_time"], reverse=True)

            closed = []
            for r in state.closed_trades:
                closed.append({
                    "ticker": r.ticker,
                    "direction": r.direction,
                    "entry_time": r.entry_time.astimezone(timezone.utc).isoformat(),
                    "exit_time": (
                        r.exit_time.astimezone(timezone.utc).isoformat()
                        if r.exit_time else None
                    ),
                    "entry_price": r.entry_price,
                    "exit_price": r.exit_price,
                    "shares": r.shares,
                    "pnl": round(r.pnl, 2) if r.pnl is not None else None,
                    "exit_reason": r.exit_reason,
                })
            closed.sort(key=lambda t: t["exit_time"] or "", reverse=True)

            cooldown = None
            if state.cooldown_until:
                cooldown = state.cooldown_until.astimezone(timezone.utc).isoformat()

            max_heat = state.config_snapshot.get("max_portfolio_heat", 0.24)
            max_pos = state.config_snapshot.get("max_open_positions", 15)

            return {
                "equity": round(state.equity, 2),
                "cash": round(state.cash, 2),
                "buying_power": round(state.buying_power, 2),
                "is_paper": state.is_paper,
                "day_pnl": day_pnl,
                "day_pnl_pct": day_pnl_pct,
                "regime_uptrend": state.regime_uptrend,
                "kill_switch_active": state.kill_switch_active,
                "portfolio_heat_pct": round(state.portfolio_heat_pct, 4),
                "max_portfolio_heat": max_heat,
                "consecutive_losses": state.consecutive_losses,
                "cooldown_until": cooldown,
                "open_positions_count": len(positions),
                "max_open_positions": max_pos,
                "positions": positions,
                "closed_trades": closed,
                "config": state.config_snapshot,
            }

    return app


def start_server(state: DashboardState) -> None:
    port = int(os.environ.get("DASHBOARD_PORT", "8080"))
    app = create_app(state)
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="error"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    logger.info("Dashboard running at http://localhost:%d", port)
```

- [ ] **Step 4: Create a stub templates.py so the import resolves**

Create `bot/dashboard/templates.py` with a minimal placeholder (will be replaced in Task 4):

```python
DASHBOARD_HTML = "<!DOCTYPE html><html><body>Dashboard loading...</body></html>"
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest testing/test_dashboard_server.py -v
```

Expected: All 8 tests pass.

- [ ] **Step 6: Commit**

```bash
git add bot/dashboard/server.py bot/dashboard/templates.py testing/test_dashboard_server.py
git commit -m "feat: add FastAPI dashboard server with /api/state endpoint"
```

---

## Task 4: HTML dashboard template

**Files:**
- Modify: `bot/dashboard/templates.py`

No unit tests — the HTML is a static string served verbatim. Visual correctness is verified by opening the browser after Task 5.

- [ ] **Step 1: Replace templates.py with the full dashboard HTML**

Overwrite `bot/dashboard/templates.py`:

```python
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>V4 Bot Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #0d1117;
      --surface: #161b22;
      --border: rgba(255,255,255,0.07);
      --text: #e2e8f0;
      --muted: #8b949e;
      --cyan: #22d3ee;
      --green: #34d399;
      --red: #f87171;
      --yellow: #facc15;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: var(--bg);
      color: var(--text);
      font-size: 13px;
      min-height: 100vh;
    }

    /* ── Header ─────────────────────────────────────────────── */
    #header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      display: flex;
      align-items: center;
      gap: 24px;
      flex-wrap: wrap;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .brand { font-size: 16px; font-weight: 700; color: #fff; letter-spacing: -0.3px; }
    .brand em { color: var(--cyan); font-style: normal; }
    .stat { display: flex; flex-direction: column; gap: 2px; }
    .stat .lbl { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.7px; }
    .stat .val { font-size: 15px; font-weight: 600; }
    .badges { margin-left: auto; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .badge {
      font-size: 10px; font-weight: 600;
      padding: 3px 10px; border-radius: 20px;
      border: 1px solid transparent;
    }
    .badge-paper  { background: rgba(34,211,238,0.10); color: var(--cyan);   border-color: rgba(34,211,238,0.20); }
    .badge-live   { background: rgba(250,204,21,0.10);  color: var(--yellow); border-color: rgba(250,204,21,0.20); }
    .badge-up     { background: rgba(52,211,153,0.10);  color: var(--green);  border-color: rgba(52,211,153,0.20); }
    .badge-down   { background: rgba(248,113,113,0.10); color: var(--red);    border-color: rgba(248,113,113,0.20); }
    .badge-active { background: rgba(52,211,153,0.10);  color: var(--green);  border-color: rgba(52,211,153,0.20); }
    .badge-halted { background: rgba(248,113,113,0.10); color: var(--red);    border-color: rgba(248,113,113,0.20); }

    /* ── Body layout ────────────────────────────────────────── */
    #body {
      display: grid;
      grid-template-columns: 1fr 240px;
      gap: 16px;
      padding: 20px 24px;
      max-width: 1400px;
    }
    .main-col, .side-col { display: flex; flex-direction: column; gap: 16px; }

    /* ── Panels ─────────────────────────────────────────────── */
    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
    }
    .panel-title {
      color: var(--muted);
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 12px;
    }

    /* ── Tables ─────────────────────────────────────────────── */
    table { width: 100%; border-collapse: collapse; }
    th {
      color: var(--muted); font-size: 10px; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.5px;
      padding: 4px 10px; text-align: left;
      border-bottom: 1px solid var(--border);
    }
    td { padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,0.03); font-size: 12px; }
    tr:last-child td { border-bottom: none; }
    .g { color: var(--green); font-weight: 600; }
    .r { color: var(--red);   font-weight: 600; }
    .m { color: var(--muted); }
    .empty-msg { color: var(--muted); font-size: 11px; text-align: center; padding: 20px 0; }

    /* ── Sidebar KV pairs ───────────────────────────────────── */
    .kv { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .kv:last-child { margin-bottom: 0; }
    .k { color: var(--muted); font-size: 11px; }
    .v { font-size: 12px; font-weight: 600; }

    /* ── Heat progress bar ──────────────────────────────────── */
    .progress-wrap { background: rgba(255,255,255,0.06); border-radius: 4px; height: 4px; margin: -2px 0 10px; }
    .progress-fill { border-radius: 4px; height: 4px; background: linear-gradient(90deg, var(--cyan), var(--green)); transition: width 0.5s; }

    /* ── Chart ──────────────────────────────────────────────── */
    #pnl-chart-wrap { height: 130px; position: relative; }
  </style>
</head>
<body>

<div id="header">
  <span class="brand">V4 <em>Bot</em></span>
  <div class="stat"><div class="lbl">Equity</div><div class="val" id="h-equity">—</div></div>
  <div class="stat"><div class="lbl">Cash</div><div class="val" id="h-cash">—</div></div>
  <div class="stat"><div class="lbl">Day P&amp;L</div><div class="val" id="h-pnl">—</div></div>
  <div class="badges">
    <span class="badge badge-paper" id="b-env">PAPER</span>
    <span class="badge badge-up"    id="b-regime">↑ UPTREND</span>
    <span class="badge badge-active" id="b-kill">✓ ACTIVE</span>
  </div>
</div>

<div id="body">
  <div class="main-col">

    <div class="panel">
      <div class="panel-title">Intraday P&amp;L</div>
      <div id="pnl-chart-wrap"><canvas id="pnl-chart"></canvas></div>
    </div>

    <div class="panel">
      <div class="panel-title">Open Positions (<span id="pos-count">0</span>)</div>
      <table>
        <thead>
          <tr>
            <th>Ticker</th><th>Shares</th><th>Entry</th><th>Stop</th>
            <th>Target</th><th>Last</th><th>Unreal. P&amp;L</th><th>Risk</th><th>Time In</th>
          </tr>
        </thead>
        <tbody id="positions-body">
          <tr><td colspan="9" class="empty-msg">No open positions</td></tr>
        </tbody>
      </table>
    </div>

    <div class="panel">
      <div class="panel-title">Closed Trades Today (<span id="trades-count">0</span>)</div>
      <table>
        <thead>
          <tr>
            <th>Time</th><th>Ticker</th><th>Dir</th><th>Entry</th>
            <th>Exit</th><th>Shares</th><th>P&amp;L</th><th>Reason</th>
          </tr>
        </thead>
        <tbody id="trades-body">
          <tr><td colspan="8" class="empty-msg">No closed trades</td></tr>
        </tbody>
      </table>
    </div>

  </div>
  <div class="side-col">

    <div class="panel">
      <div class="panel-title">Portfolio Risk</div>
      <div class="kv"><span class="k">Heat</span><span class="v" id="s-heat">—</span></div>
      <div class="progress-wrap"><div class="progress-fill" id="heat-bar" style="width:0%"></div></div>
      <div class="kv"><span class="k">Positions</span><span class="v" id="s-pos">—</span></div>
      <div class="kv"><span class="k">Consec. Losses</span><span class="v" id="s-losses">—</span></div>
      <div class="kv"><span class="k">Cooldown</span><span class="v m" id="s-cooldown">—</span></div>
    </div>

    <div class="panel">
      <div class="panel-title">Config</div>
      <div id="config-body"><span class="m" style="font-size:11px">Loading...</span></div>
    </div>

  </div>
</div>

<script>
// ── Chart setup ──────────────────────────────────────────────
const ctx = document.getElementById('pnl-chart').getContext('2d');
const pnlChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Cumulative P&L',
      data: [],
      borderColor: '#22d3ee',
      backgroundColor: 'rgba(34,211,238,0.06)',
      borderWidth: 2,
      pointRadius: 3,
      pointBackgroundColor: '#22d3ee',
      fill: true,
      tension: 0.3,
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: { label: c => '$' + c.parsed.y.toFixed(2) }
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#8b949e', font: { size: 10 } }
      },
      y: {
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#8b949e', font: { size: 10 }, callback: v => '$' + v.toFixed(0) }
      }
    }
  }
});

// ── Helpers ──────────────────────────────────────────────────
const fmt2 = n => n == null ? '—' : Math.abs(n).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
const timeFmt = iso => iso ? iso.substring(11, 16) : '—';
const timeIn = iso => {
  const mins = Math.floor((Date.now() - new Date(iso)) / 60000);
  return mins < 60 ? mins + 'm' : Math.floor(mins / 60) + 'h ' + (mins % 60) + 'm';
};
const sign = n => n >= 0 ? '+' : '-';

// ── Updaters ─────────────────────────────────────────────────
function updateHeader(d) {
  const approxEquity = d.equity + d.day_pnl;
  document.getElementById('h-equity').textContent = '$' + fmt2(approxEquity);
  document.getElementById('h-cash').textContent = '$' + fmt2(d.cash);

  const pnlEl = document.getElementById('h-pnl');
  pnlEl.textContent = sign(d.day_pnl) + '$' + fmt2(d.day_pnl) + ' (' + sign(d.day_pnl_pct) + (Math.abs(d.day_pnl_pct) * 100).toFixed(2) + '%)';
  pnlEl.className = 'val ' + (d.day_pnl >= 0 ? 'g' : 'r');

  const bEnv = document.getElementById('b-env');
  bEnv.textContent = d.is_paper ? 'PAPER' : 'LIVE';
  bEnv.className = 'badge ' + (d.is_paper ? 'badge-paper' : 'badge-live');

  const bReg = document.getElementById('b-regime');
  bReg.textContent = d.regime_uptrend ? '↑ UPTREND' : '↓ BLOCKED';
  bReg.className = 'badge ' + (d.regime_uptrend ? 'badge-up' : 'badge-down');

  const bKill = document.getElementById('b-kill');
  bKill.textContent = d.kill_switch_active ? '✗ HALTED' : '✓ ACTIVE';
  bKill.className = 'badge ' + (d.kill_switch_active ? 'badge-halted' : 'badge-active');
}

function updatePositions(d) {
  document.getElementById('pos-count').textContent = d.open_positions_count;
  const tbody = document.getElementById('positions-body');
  if (!d.positions.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty-msg">No open positions</td></tr>';
    return;
  }
  tbody.innerHTML = d.positions.map(p => {
    const cls = p.unrealized_pnl >= 0 ? 'g' : 'r';
    return '<tr>' +
      '<td><strong>' + p.ticker + '</strong></td>' +
      '<td>' + p.shares + '</td>' +
      '<td>$' + fmt2(p.entry_price) + '</td>' +
      '<td>$' + fmt2(p.stop_price) + '</td>' +
      '<td>$' + fmt2(p.target_price) + '</td>' +
      '<td>$' + fmt2(p.last_price) + '</td>' +
      '<td class="' + cls + '">' + sign(p.unrealized_pnl) + '$' + fmt2(p.unrealized_pnl) + '</td>' +
      '<td class="m">$' + fmt2(p.open_risk) + '</td>' +
      '<td class="m">' + timeIn(p.entry_time) + '</td>' +
      '</tr>';
  }).join('');
}

function updateTrades(d) {
  document.getElementById('trades-count').textContent = d.closed_trades.length;
  const tbody = document.getElementById('trades-body');
  if (!d.closed_trades.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty-msg">No closed trades</td></tr>';
    return;
  }
  tbody.innerHTML = d.closed_trades.map(t => {
    const cls = (t.pnl || 0) >= 0 ? 'g' : 'r';
    return '<tr>' +
      '<td class="m">' + timeFmt(t.exit_time) + '</td>' +
      '<td><strong>' + t.ticker + '</strong></td>' +
      '<td class="m">' + t.direction + '</td>' +
      '<td>$' + fmt2(t.entry_price) + '</td>' +
      '<td>$' + fmt2(t.exit_price) + '</td>' +
      '<td>' + t.shares + '</td>' +
      '<td class="' + cls + '">' + sign(t.pnl || 0) + '$' + fmt2(t.pnl) + '</td>' +
      '<td class="m">' + (t.exit_reason || '—') + '</td>' +
      '</tr>';
  }).join('');
}

function updateChart(d) {
  const sorted = d.closed_trades
    .filter(t => t.exit_time && t.pnl != null)
    .slice()
    .sort((a, b) => a.exit_time.localeCompare(b.exit_time));
  let cum = 0;
  const labels = [], values = [];
  for (const t of sorted) {
    cum += t.pnl;
    labels.push(timeFmt(t.exit_time));
    values.push(parseFloat(cum.toFixed(2)));
  }
  pnlChart.data.labels = labels;
  pnlChart.data.datasets[0].data = values;
  const color = cum >= 0 ? '#22d3ee' : '#f87171';
  pnlChart.data.datasets[0].borderColor = color;
  pnlChart.data.datasets[0].backgroundColor = cum >= 0 ? 'rgba(34,211,238,0.06)' : 'rgba(248,113,113,0.06)';
  pnlChart.update('none');
}

function updateRisk(d) {
  const hPct = (d.portfolio_heat_pct * 100).toFixed(1);
  const maxPct = (d.max_portfolio_heat * 100).toFixed(1);
  document.getElementById('s-heat').textContent = hPct + '% / ' + maxPct + '%';
  const fill = Math.min((d.portfolio_heat_pct / (d.max_portfolio_heat || 1)) * 100, 100);
  document.getElementById('heat-bar').style.width = fill + '%';
  document.getElementById('s-pos').textContent = d.open_positions_count + ' / ' + d.max_open_positions;
  document.getElementById('s-losses').textContent = d.consecutive_losses;

  const cdEl = document.getElementById('s-cooldown');
  if (d.cooldown_until) {
    const secs = Math.max(0, Math.floor((new Date(d.cooldown_until) - Date.now()) / 1000));
    cdEl.textContent = Math.floor(secs / 60) + ':' + String(secs % 60).padStart(2, '0');
    cdEl.className = 'v r';
  } else {
    cdEl.textContent = '—';
    cdEl.className = 'v m';
  }
}

function updateConfig(d) {
  const c = d.config;
  if (!c || !Object.keys(c).length) return;
  const rows = [
    ['Risk/Trade', (c.risk_per_trade * 100).toFixed(1) + '%'],
    ['Max Heat', (c.max_portfolio_heat * 100).toFixed(0) + '%'],
    ['Min RelVol', c.stage2_min_relative_volume + '×'],
    ['Min Δ Price', (c.stage1_min_price_change_pct * 100).toFixed(0) + '%'],
    ['Buy Pressure', '≥ ' + (c.stage2_buying_pressure_min * 100).toFixed(0) + '%'],
    ['EOD Exit', c.eod_evaluation],
    ['Conf. Tiers', c.confidence_tiers],
  ];
  document.getElementById('config-body').innerHTML = rows.map(([k, v]) =>
    '<div class="kv"><span class="k">' + k + '</span><span class="v">' + v + '</span></div>'
  ).join('');
}

// ── Poll loop ────────────────────────────────────────────────
async function poll() {
  try {
    const r = await fetch('/api/state');
    if (!r.ok) return;
    const d = await r.json();
    updateHeader(d);
    updatePositions(d);
    updateTrades(d);
    updateChart(d);
    updateRisk(d);
    updateConfig(d);
  } catch (e) {
    console.error('Dashboard poll error:', e);
  }
}

poll();
setInterval(poll, 3000);
</script>
</body>
</html>"""
```

- [ ] **Step 2: Run the server tests again to confirm the index still passes**

```bash
pytest testing/test_dashboard_server.py::test_index_returns_html -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add bot/dashboard/templates.py
git commit -m "feat: add dashboard HTML template — Slate Dark + Cyan theme"
```

---

## Task 5: Wire into bot/main.py

**Files:**
- Modify: `bot/main.py`

- [ ] **Step 1: Add imports at the top of bot/main.py**

After the existing imports block (after `from bot.trade_logger import TradeLogger`), add:

```python
from bot.dashboard.server import start_server
from bot.dashboard.state import DashboardState
```

- [ ] **Step 2: Add _sync_portfolio helper before main()**

Add this function just before the `def main()` line:

```python
def _sync_portfolio(dash: DashboardState, portfolio: PortfolioState) -> None:
    with dash._lock:
        dash.positions = dict(portfolio.positions)
        dash.portfolio_heat_pct = portfolio.portfolio_heat_pct
        dash.kill_switch_active = portfolio.kill_switch_active
        dash.consecutive_losses = portfolio.consecutive_losses
        dash.cooldown_until = portfolio.cooldown_until
```

- [ ] **Step 3: Initialize DashboardState in main() after account info is fetched**

In `main()`, after these existing lines:
```python
account = broker.get_account_info()
equity = account["portfolio_value"]
```

Add:
```python
dash = DashboardState()
dash.equity = equity
dash.cash = account["cash"]
dash.buying_power = account["buying_power"]
dash.is_paper = broker._is_paper
dash.config_snapshot = {
    "risk_per_trade": config.risk_per_trade,
    "max_portfolio_heat": config.max_portfolio_heat,
    "max_open_positions": config.max_open_positions,
    "stage2_min_relative_volume": config.stage2_min_relative_volume,
    "stage1_min_price_change_pct": config.stage1_min_price_change_pct,
    "stage2_buying_pressure_min": config.stage2_buying_pressure_min,
    "eod_evaluation": config.eod_evaluation,
    "confidence_tiers": (
        f"{config.confidence_tier1_multiplier:.0f}×"
        f"/{config.confidence_tier2_multiplier:.0f}×"
        f"/{config.confidence_tier3_multiplier:.0f}×"
        f"/{config.confidence_tier4_multiplier:.0f}×"
    ),
}
```

- [ ] **Step 4: Start the server before stream.run()**

In `main()`, after `scanner_thread.start()` and before `stream.run()`, add:

```python
start_server(dash)
```

- [ ] **Step 5: Update last_prices and kill switch state on every bar**

In `on_bar`, replace:
```python
    kill_switch.check(portfolio, now)
    if portfolio.kill_switch_active:
        return
```

With:
```python
    kill_switch.check(portfolio, now)
    with dash._lock:
        dash.last_prices[bar.symbol] = bar.close
        dash.kill_switch_active = portfolio.kill_switch_active
        dash.consecutive_losses = portfolio.consecutive_losses
        dash.cooldown_until = portfolio.cooldown_until
    if portfolio.kill_switch_active:
        return
```

- [ ] **Step 6: Sync portfolio state after the EOD exit**

In `on_bar`, find the EOD exit block. Replace:
```python
                if record:
                    _close_record(record, now, bar.close, "eod", trade_logger)
                return
```

With:
```python
                if record:
                    _close_record(record, now, bar.close, "eod", trade_logger)
                    with dash._lock:
                        dash.closed_trades.append(record)
                _sync_portfolio(dash, portfolio)
                return
```

- [ ] **Step 7: Sync portfolio state after intraday exit**

In `on_bar`, find the intraday exit block inside `if instruction:`. Replace:
```python
                if instruction:
                    exit_price = instruction.limit_price if instruction.limit_price else bar.close
                    _execute_exit(instruction, position)
                    portfolio.remove_position(bar.symbol)
                    record = open_records.pop(bar.symbol, None)
                    if record:
                        _close_record(record, now, exit_price, instruction.reason, trade_logger)
```

With:
```python
                if instruction:
                    exit_price = instruction.limit_price if instruction.limit_price else bar.close
                    _execute_exit(instruction, position)
                    portfolio.remove_position(bar.symbol)
                    record = open_records.pop(bar.symbol, None)
                    if record:
                        _close_record(record, now, exit_price, instruction.reason, trade_logger)
                        with dash._lock:
                            dash.closed_trades.append(record)
                    _sync_portfolio(dash, portfolio)
```

- [ ] **Step 8: Sync regime state when it changes**

In `on_bar`, find the regime update block. Replace:
```python
        if _regime["date"] != today:
            _regime["uptrend"] = regime_filter.is_uptrend(today)
            _regime["date"] = today
            if _regime["uptrend"]:
                logger.info("REGIME %s: SPY uptrend — long entries enabled", today)
            else:
                logger.info("REGIME %s: SPY below 20-day MA — long entries blocked", today)
```

With:
```python
        if _regime["date"] != today:
            _regime["uptrend"] = regime_filter.is_uptrend(today)
            _regime["date"] = today
            with dash._lock:
                dash.regime_uptrend = _regime["uptrend"]
                dash.regime_date = today
            if _regime["uptrend"]:
                logger.info("REGIME %s: SPY uptrend — long entries enabled", today)
            else:
                logger.info("REGIME %s: SPY below 20-day MA — long entries blocked", today)
```

- [ ] **Step 9: Sync portfolio state after entry**

In `on_bar`, find the line `portfolio.add_position(position)`. Add immediately after it:

```python
            _sync_portfolio(dash, portfolio)
```

- [ ] **Step 10: Run the full test suite**

```bash
pytest testing/ -v --ignore=testing/test_broker.py --ignore=testing/test_alpaca_data.py --ignore=testing/test_yfinance_data.py -x
```

Expected: All tests pass (skipping tests that need live API credentials).

- [ ] **Step 11: Commit**

```bash
git add bot/main.py
git commit -m "feat: wire DashboardState into bot main loop — syncs positions, trades, regime on every change"
```

---

## Task 6: Smoke test in browser

- [ ] **Step 1: Start the bot**

```bash
python3 -m bot.main
```

Expected log line: `INFO bot.dashboard Dashboard running at http://localhost:8080`

- [ ] **Step 2: Open the dashboard**

Open `http://localhost:8080` in a browser.

Expected: Dashboard loads with the Slate Dark + Cyan theme. Header shows equity, cash, Day P&L. Risk sidebar shows heat % and position count. Config panel shows bot parameters. Positions and trades tables show empty state messages.

- [ ] **Step 3: Verify live updates**

Wait for any bar to arrive. Check the browser — the positions and risk panels should update within 3 seconds.

- [ ] **Step 4: Add .superpowers to .gitignore**

```bash
echo ".superpowers/" >> .gitignore
git add .gitignore
git commit -m "chore: ignore .superpowers brainstorm session files"
```
