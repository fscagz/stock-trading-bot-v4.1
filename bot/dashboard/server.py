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
                is_short = getattr(pos, "direction", "long") == "short"
                if is_short:
                    unrealized = round((pos.entry_price - last_price) * pos.shares, 2)
                else:
                    unrealized = round((last_price - pos.entry_price) * pos.shares, 2)
                positions.append({
                    "ticker": ticker,
                    "direction": getattr(pos, "direction", "long"),
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

            short_max_heat = state.short_config_snapshot.get("max_portfolio_heat", 0.06)

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
                "long_strategy_name": state.long_strategy_name,
                "short_enabled": state.short_enabled,
                "short_allowed": state.short_allowed,
                "short_heat_pct": round(state.short_heat_pct, 4),
                "short_max_heat": short_max_heat,
                "short_config": state.short_config_snapshot,
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
