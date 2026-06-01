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
