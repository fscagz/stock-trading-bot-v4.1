# Trading Bot Dashboard — Design Spec
**Date:** 2026-06-01  
**Status:** Approved

---

## Overview

A web dashboard that runs alongside the live trading bot (`python3 -m bot.main`), displaying real-time account status, open positions, trade history, risk metrics, and configuration. The bot remains unchanged in its trading logic; the dashboard is purely read-only and shares in-memory state.

---

## Architecture

### Approach: FastAPI in a background thread, shared in-memory state

The bot starts a FastAPI server in a daemon thread at startup. A single `DashboardState` object is mutated by `on_bar` (and on trade close) and read by the API handler. The browser polls `/api/state` every 3 seconds.

```
on_bar() ──mutates──► DashboardState ◄──reads── GET /api/state ◄──polls── Browser
```

This approach was chosen over calling the Alpaca REST API from the dashboard to avoid adding API calls that could hit rate limits during live trading.

### No changes to trading logic

- No modifications to order submission, position management, or risk checks
- `TradeLogger` CSV output is unaffected
- The dashboard is additive only

---

## New Files

### `bot/dashboard/__init__.py`
Empty package marker.

### `bot/dashboard/state.py`
Thread-safe shared state object.

```python
@dataclass
class DashboardState:
    # Account
    equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    is_paper: bool = True
    session_start_equity: float = 0.0

    # Portfolio
    positions: Dict[str, Position] = field(default_factory=dict)
    portfolio_heat_pct: float = 0.0
    kill_switch_active: bool = False
    consecutive_losses: int = 0
    cooldown_until: Optional[datetime] = None

    # Regime
    regime_uptrend: bool = True
    regime_date: Optional[date] = None

    # Trades closed today
    closed_trades: List[TradeRecord] = field(default_factory=list)

    # Last bar close price per symbol (for unrealized P&L)
    last_prices: Dict[str, float] = field(default_factory=dict)

    # Config snapshot (populated once at startup)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
```

All mutations are wrapped with `state._lock`. The lock is held only for dict/list operations, never across I/O.

### `bot/dashboard/server.py`
FastAPI application with two routes:

- `GET /` — serves `index.html` (embedded as a string constant, no static file serving needed)
- `GET /api/state` — returns a JSON snapshot built from `DashboardState`

The JSON shape returned by `/api/state`:

```json
{
  "equity": 48231.00,
  "cash": 31450.00,
  "buying_power": 62900.00,
  "is_paper": true,
  "day_pnl": 312.40,
  "day_pnl_pct": 0.0065,
  "regime_uptrend": true,
  "kill_switch_active": false,
  "portfolio_heat_pct": 0.082,
  "max_portfolio_heat": 0.24,
  "consecutive_losses": 0,
  "cooldown_until": null,
  "open_positions_count": 3,
  "max_open_positions": 15,
  "positions": [
    {
      "ticker": "ASTC",
      "shares": 420,
      "entry_price": 4.21,
      "stop_price": 3.90,
      "target_price": 4.84,
      "last_price": 4.52,
      "unrealized_pnl": 130.20,
      "open_risk": 130.20,
      "entry_time": "2026-06-01T09:32:00Z"
    }
  ],
  "closed_trades": [
    {
      "ticker": "MSOX",
      "direction": "long",
      "entry_time": "...",
      "exit_time": "...",
      "entry_price": 5.14,
      "exit_price": 5.61,
      "shares": 120,
      "pnl": 56.40,
      "exit_reason": "target"
    }
  ],
  "config": {
    "risk_per_trade": 0.04,
    "max_portfolio_heat": 0.24,
    "stage2_min_relative_volume": 10.0,
    "stage1_min_price_change_pct": 0.15,
    "stage2_buying_pressure_min": 0.85,
    "eod_evaluation": "15:25",
    "confidence_tiers": "1×/2×/4×/8×"
  }
}
```

The server is started with `uvicorn` in a background daemon thread. Default port: `8080`, overridable via `DASHBOARD_PORT` env var.

### `bot/dashboard/templates.py`
Holds the dashboard HTML as a Python string constant (`DASHBOARD_HTML`). This avoids any static file serving and keeps deployment simple. The HTML uses:
- **Chart.js** (CDN) for the intraday P&L bar chart
- Vanilla JS `fetch` polling `/api/state` every 3 seconds
- Inline CSS — Slate Dark + Cyan theme (approved in design)

---

## Modified Files

### `bot/main.py`
Three additions, no logic changes:

1. **Import and instantiate** `DashboardState` and start the server thread before `stream.run()`.
2. **Update `last_prices`** inside `on_bar` on every bar for symbols in the portfolio (needed for unrealized P&L).
3. **Sync state** to `DashboardState` after every position add/remove and after every trade close: positions dict, heat %, kill switch flag, consecutive losses, regime, and newly closed `TradeRecord`.

---

## Dashboard UI — Sections

### Header bar
Equity · Cash · Daily P&L ($ and %) · PAPER/LIVE badge · Regime badge (↑ UPTREND / ↓ BLOCKED) · Kill switch badge (✓ ACTIVE / ✗ HALTED)

### Intraday P&L chart
Bar chart showing cumulative P&L after each closed trade. X-axis = trade close time, Y-axis = cumulative $. Rendered with Chart.js. Updates when `/api/state` returns new closed trades.

### Open Positions table
Columns: Ticker · Shares · Entry · Stop · Target · Last Price · Unrealized P&L · Open Risk · Time In  
Sorted by entry time descending.  
Unrealized P&L = `(last_price - entry_price) * shares` (long positions only; last_price from most recent bar seen by `on_bar`).

### Closed Trades log
Columns: Time · Ticker · Direction · Entry → Exit · Shares · P&L · Exit Reason  
Sorted by exit time descending. P&L colored green/red.

### Risk sidebar
- Portfolio heat % with gradient progress bar (vs. max)
- Open positions count (vs. max)
- Consecutive losses
- Cooldown timer (countdown if active, "—" otherwise)

### Config snapshot sidebar
Displays the key config fields from `config_snapshot`: risk/trade, max heat, min relative volume, min price change %, buying pressure range, EOD exit time, confidence tier multipliers.

---

## Launch

No new command. `python3 -m bot.main` starts both the bot and the dashboard server. On startup, a log line prints the URL:

```
INFO  bot.dashboard  Dashboard running at http://localhost:8080
```

Port is configurable via `DASHBOARD_PORT` environment variable.

---

## Dependencies

Add to `requirements.txt`:
- `fastapi`
- `uvicorn`

Both are lightweight and have no conflict with existing dependencies.

---

## Out of Scope

- Authentication / access control (single-user local tool)
- Historical data across multiple days
- Controls to start/stop/modify the bot from the UI
- WebSocket push (polling every 3s is sufficient for 1-min bars)
