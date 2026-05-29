from datetime import date, datetime
from zoneinfo import ZoneInfo
from typing import List

from bot.backtest.simulator import Simulator
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


# Timeline: bar0(9:30) gets atr=None → validator skipped.
# Validator accumulates from bar1 onward. With lookback=2, signal needs 3 bars in
# validator history → bar1+bar2+bar3 → signal at bar3 (9:33), fill at bar4 (9:34).
def _lead_bars(sym: str) -> List[Bar]:
    return [
        _bar(sym, 9, 30, 2.00, 2.10, 1.98, 2.05),  # bar0: atr=None, validator skipped
        _bar(sym, 9, 31, 2.05, 2.20, 2.03, 2.15),  # bar1: validator len=1 < 3 → False
        _bar(sym, 9, 32, 2.15, 2.30, 2.13, 2.25),  # bar2: validator len=2 < 3 → False
        _bar(sym, 9, 33, 2.25, 2.65, 2.24, 2.58),  # bar3: SIGNAL (RoC=20%, BP=83%, vol=2.5x)
    ]


_FILL_OPEN = 2.58   # bar4's open — the price the pending entry fills at
_FILL_BAR = _bar("ASTC", 9, 34, _FILL_OPEN, 2.70, 2.56, 2.65)

# baseline 200k/min → bar volume 500k = 2.5x, satisfies min_relative_volume=2.0
_BASELINE = {"ASTC": 200_000.0}


def test_entry_fills_at_next_bar_open_plus_slippage():
    sim = _make_sim(slippage=0.01)
    # EOD bar: low volume → should_hold_overnight=False → EOD close
    eod_bar = _bar("ASTC", 15, 25, 2.65, 2.70, 2.62, 2.67, v=30_000)
    result = sim.run_day(_DATE, {"ASTC": _lead_bars("ASTC") + [_FILL_BAR, eod_bar]}, _BASELINE)

    assert len(result.trades) == 1
    expected_fill = round(_FILL_OPEN * 1.01, 2)  # 2.58 * 1.01 = 2.61
    assert result.trades[0].entry_price == expected_fill


def test_stop_hit_fills_at_stop_price():
    sim = _make_sim(slippage=0.0)
    # crash_bar: low=0.50 is well below any computed stop
    crash_bar = _bar("ASTC", 9, 35, 2.65, 2.66, 0.50, 2.30, v=100_000)
    result = sim.run_day(_DATE, {"ASTC": _lead_bars("ASTC") + [_FILL_BAR, crash_bar]}, _BASELINE)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "hard_stop"
    assert trade.exit_price == trade.stop_price


def test_eod_close_at_1525():
    sim = _make_sim(slippage=0.0)
    # low volume triggers should_hold_overnight=False → close at eod_bar.close
    eod_bar = _bar("ASTC", 15, 25, 2.65, 2.70, 2.62, 2.35, v=30_000)
    result = sim.run_day(_DATE, {"ASTC": _lead_bars("ASTC") + [_FILL_BAR, eod_bar]}, _BASELINE)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "eod"
    assert trade.exit_price == 2.35


def test_no_entry_when_no_next_bar():
    """Signal fires on bar3 but there is no bar4 to fill → no trade."""
    sim = _make_sim()
    result = sim.run_day(_DATE, {"ASTC": _lead_bars("ASTC")}, _BASELINE)
    assert len(result.trades) == 0


def test_equity_adjusts_on_closed_trade():
    sim = _make_sim(equity=10_000.0, slippage=0.0)
    eod_bar = _bar("ASTC", 15, 25, 2.65, 2.70, 2.62, 2.67, v=30_000)
    result = sim.run_day(_DATE, {"ASTC": _lead_bars("ASTC") + [_FILL_BAR, eod_bar]}, _BASELINE)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.pnl is not None
    assert len(result.equity_curve) > 0
