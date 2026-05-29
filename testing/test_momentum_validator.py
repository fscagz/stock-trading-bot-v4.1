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
    # 6th bar: +8% price acceleration, 5× volume, closes near high
    bar = _bar("ASTC", 2.25, 2.28, 2.05, 500_000)
    # roc = (2.25 - 2.08) / 2.08 = ~8%  ✓
    # rel_vol = 500k / 100k = 5×  ✓
    # buying pressure: range=0.23, top 25% starts at 2.225, close=2.25  ✓
    result = v.validate(bar, baseline_volume_per_min=100_000)
    assert result is True
