from datetime import datetime, timezone
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
    bar = _bar(close=2.50, volume=300_000)
    mgr.on_bar(bar, pos, vwap=2.30, baseline_volume_per_min=50_000)
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
    # Close below VWAP on elevated volume (200k vs 50k baseline = 4× >= 2×)
    bar = _bar(close=1.85, volume=200_000)
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
