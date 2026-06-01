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
