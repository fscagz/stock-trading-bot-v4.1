from bot.config import V4Config


def test_v4config_defaults():
    cfg = V4Config()
    assert cfg.max_sector_positions == 15
    assert cfg.min_price == 0.50
    assert cfg.max_spread_pct == 0.01
    assert cfg.scanner_interval_seconds == 30
    assert cfg.stage1_min_price_change_pct == 0.05
    assert cfg.stage2_roc_min_pct == 0.03
    assert cfg.stage2_min_relative_volume == 4.0
    assert cfg.stage2_buying_pressure_min == 0.75
    assert cfg.trailing_stop_atr_multiple == 2.0


def test_v4config_is_compatible_with_intraday_config():
    from bot.intraday.config import IntradayConfig
    cfg = V4Config()
    assert isinstance(cfg, IntradayConfig)


from bot.intraday.types import Position
from datetime import datetime, timezone


def test_position_has_v4_fields():
    pos = Position(
        ticker="ASTC",
        direction="long",
        shares=100,
        entry_price=2.00,
        stop_price=1.70,
        target_price=2.90,
        entry_time=datetime.now(timezone.utc),
        atr_at_entry=0.20,
        signals=["momentum"],
        sector="Unknown",
        highest_close=2.00,
        stop_order_id="abc123",
        entry_bar_volume=500_000,
    )
    assert pos.highest_close == 2.00
    assert pos.stop_order_id == "abc123"
    assert pos.entry_bar_volume == 500_000
