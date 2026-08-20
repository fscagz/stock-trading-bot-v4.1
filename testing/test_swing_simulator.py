from datetime import date

import pandas as pd
import pytest

from bot.backtest.swing_simulator import (
    EventResult, SwingConfig, SwingEvent, run_portfolio, simulate_event,
)


def _daily(rows):
    """rows: list of (date_str, o, h, l, c). Volume constant."""
    idx = pd.DatetimeIndex([r[0] for r in rows], name="date")
    return pd.DataFrame(
        {"open": [r[1] for r in rows], "high": [r[2] for r in rows],
         "low": [r[3] for r in rows], "close": [r[4] for r in rows],
         "volume": [1_000_000] * len(rows)},
        index=idx,
    )


def _cfg(**kw):
    defaults = dict(entry_slippage_pct=0.0, exit_slippage_pct=0.0,
                    scale_out_r=2.0, scale_out_frac=0.5,
                    trail_ma_days=3, max_hold_days=20)
    defaults.update(kw)
    return SwingConfig(**defaults)


def _minute_bars(specs):
    """specs: list of (minutes_after_open, o, h, l, c)."""
    out = []
    for m, o, h, l, c in specs:
        hh, mm = 13 + (30 + m) // 60, (30 + m) % 60
        out.append({"t": f"2026-01-05T{hh:02d}:{mm:02d}:00Z",
                    "o": o, "h": h, "l": l, "c": c, "v": 10_000})
    return out


EVENT = SwingEvent(symbol="TEST", day=date(2026, 1, 5), gap_pct=0.30)


def test_close_green_entry_and_gap_stop_fills_at_open():
    daily = _daily([
        ("2026-01-05", 10.0, 12.0, 9.5, 11.0),   # event day: green close
        ("2026-01-06", 9.0, 9.2, 8.8, 9.0),      # opens below 9.5 stop → fill at open
    ])
    res = simulate_event(EVENT, daily, _cfg(), entry_mode="close_green")
    assert res.entered
    assert res.entry_price == pytest.approx(11.0)
    assert res.stop_price == pytest.approx(9.5)
    assert res.exits[0][1] == pytest.approx(9.0)      # open, not stop
    assert res.exits[0][3] == "gap_stop"
    assert res.r_multiple == pytest.approx((9.0 - 11.0) / 1.5)


def test_close_green_skips_red_close():
    daily = _daily([("2026-01-05", 10.0, 12.0, 9.5, 9.8)])
    res = simulate_event(EVENT, daily, _cfg(), entry_mode="close_green")
    assert not res.entered
    assert res.skip_reason == "closed_red"


def test_scale_out_breakeven_then_trail_exit():
    # entry 11, stop 9.5 → risk 1.5; scale level = 11 + 3 = 14
    daily = _daily([
        ("2026-01-05", 10.0, 12.0, 9.5, 11.0),
        ("2026-01-06", 11.5, 14.5, 11.2, 14.0),   # hits 14 → scale half, stop → 11
        ("2026-01-07", 14.0, 15.0, 13.5, 14.5),
        ("2026-01-08", 14.0, 14.2, 13.0, 13.0),   # close 13 < SMA3 of (14,14.5,13)=13.83 → trail exit
    ])
    res = simulate_event(EVENT, daily, _cfg(), entry_mode="close_green")
    assert [x[3] for x in res.exits] == ["scale_out", "trail_ma"]
    assert res.exits[0][1] == pytest.approx(14.0)
    assert res.exits[0][2] == pytest.approx(0.5)
    assert res.exits[1][1] == pytest.approx(13.0)
    # R = (0.5×(14−11) + 0.5×(13−11)) / 1.5
    assert res.r_multiple == pytest.approx((0.5 * 3 + 0.5 * 2) / 1.5)


def test_breakeven_stop_protects_after_scale_out():
    daily = _daily([
        ("2026-01-05", 10.0, 12.0, 9.5, 11.0),
        ("2026-01-06", 11.5, 14.5, 11.2, 14.0),   # scale at 14, stop → 11
        ("2026-01-07", 13.0, 13.5, 10.5, 10.6),   # low 10.5 ≤ 11 → stopped at 11
    ])
    res = simulate_event(EVENT, daily, _cfg(), entry_mode="close_green")
    assert [x[3] for x in res.exits] == ["scale_out", "stop"]
    assert res.exits[1][1] == pytest.approx(11.0)
    assert res.r_multiple == pytest.approx((0.5 * 3 + 0.5 * 0) / 1.5)


def test_orb_entry_break_of_opening_range():
    bars = _minute_bars([
        (0, 10.0, 10.5, 9.8, 10.2),
        (15, 10.2, 10.6, 10.0, 10.4),
        (31, 10.4, 10.8, 10.3, 10.7),   # breaks OR high 10.6 → entry at 10.6
        (60, 10.7, 11.0, 10.5, 10.9),
    ])
    daily = _daily([
        ("2026-01-05", 10.0, 11.0, 9.8, 10.9),
        ("2026-01-06", 11.0, 12.0, 10.8, 11.8),
    ])
    res = simulate_event(EVENT, daily, _cfg(max_hold_days=1), bars, entry_mode="orb")
    assert res.entered
    assert res.entry_price == pytest.approx(10.6)
    assert res.stop_price == pytest.approx(9.8)   # low of day at entry
    assert res.exits[-1][3] == "time"


def test_orb_no_break_no_trade():
    bars = _minute_bars([
        (0, 10.0, 10.5, 9.8, 10.2),
        (31, 10.2, 10.4, 10.0, 10.1),
        (60, 10.1, 10.3, 9.9, 10.0),
    ])
    daily = _daily([("2026-01-05", 10.0, 10.5, 9.8, 10.0)])
    res = simulate_event(EVENT, daily, _cfg(), bars, entry_mode="orb")
    assert not res.entered
    assert res.skip_reason == "no_orb_break"


def test_orb_same_day_stop():
    bars = _minute_bars([
        (0, 10.0, 10.5, 9.8, 10.2),
        (31, 10.4, 10.8, 10.3, 10.7),   # entry at 10.6, stop 9.8
        (90, 10.0, 10.1, 9.5, 9.6),     # low 9.5 ≤ 9.8 → stopped same day
    ])
    daily = _daily([("2026-01-05", 10.0, 10.8, 9.5, 9.6)])
    res = simulate_event(EVENT, daily, _cfg(), bars, entry_mode="orb")
    assert res.entered
    assert res.exits[0][3] == "stop_same_day"
    assert res.r_multiple == pytest.approx(-1.0)


def test_portfolio_respects_position_cap_and_compounds():
    def mk(sym, entry_d, exit_d, r):
        ev = SwingEvent(symbol=sym, day=entry_d, gap_pct=0.3)
        res = EventResult(event=ev, entered=True, entry_date=entry_d,
                          entry_price=10.0, stop_price=9.0,
                          exits=[(exit_d, 10.0 + r, 1.0, "stop")])
        res.r_multiple = r
        return res

    results = [
        mk("AAA", date(2026, 1, 5), date(2026, 1, 9), +2.0),
        mk("BBB", date(2026, 1, 6), date(2026, 1, 9), -1.0),
        mk("CCC", date(2026, 1, 7), date(2026, 1, 9), +1.0),  # blocked: cap=2
        mk("DDD", date(2026, 1, 12), date(2026, 1, 14), +1.0),
    ]
    cfg = SwingConfig(max_positions=2, risk_per_trade=0.01, max_notional_pct=10.0)
    trades, curve = run_portfolio(results, cfg, initial_equity=100_000)
    assert [t.symbol for t in trades] == ["AAA", "BBB", "DDD"]
    # AAA: risk 1000 → +2000; BBB: risk 1000 → −1000; DDD risk 1.01×1000 → +1010
    assert trades[0].pnl == pytest.approx(2000, rel=1e-3)
    assert trades[1].pnl == pytest.approx(-1000, rel=1e-3)
    assert curve[-1][1] == pytest.approx(102_010, rel=1e-3)


def test_portfolio_no_duplicate_symbol_while_open():
    def mk(sym, entry_d, exit_d):
        ev = SwingEvent(symbol=sym, day=entry_d, gap_pct=0.3)
        res = EventResult(event=ev, entered=True, entry_date=entry_d,
                          entry_price=10.0, stop_price=9.0,
                          exits=[(exit_d, 11.0, 1.0, "stop")])
        res.r_multiple = 1.0
        return res

    results = [
        mk("AAA", date(2026, 1, 5), date(2026, 1, 20)),
        mk("AAA", date(2026, 1, 7), date(2026, 1, 21)),
    ]
    trades, _ = run_portfolio(results, SwingConfig(max_positions=5), 100_000)
    assert len(trades) == 1
