from __future__ import annotations
from dataclasses import dataclass
from bot.intraday.config import IntradayConfig


@dataclass
class V4Config(IntradayConfig):
    # --- V4 overrides of V3 defaults ---
    max_open_positions: int = 15
    max_sector_positions: int = 15
    min_price: float = 0.50
    max_price: float = float("inf")
    max_spread_pct: float = 0.01

    # --- Trailing stop ---
    trailing_stop_atr_multiple: float = 2.0

    # --- Scanner ---
    scanner_interval_seconds: int = 30
    scanner_top_n: int = 50

    # --- Stage 1 filter ---
    stage1_min_price_change_pct: float = 0.05
    stage1_min_price: float = 0.50

    # --- Stage 2 momentum validation ---
    stage2_roc_lookback_bars: int = 5
    stage2_roc_min_pct: float = 0.03
    stage2_min_relative_volume: float = 4.0
    stage2_buying_pressure_min: float = 0.75

    # --- Exit thresholds ---
    vwap_break_volume_ratio: float = 2.0
    volume_collapse_ratio: float = 0.5
    structure_break_bars: int = 2
    overnight_min_volume_ratio: float = 2.0

    # --- Overnight hold evaluation time (ET) ---
    eod_evaluation: str = "15:25"

    # --- Entry distance filter ---
    stage2_min_dist_from_day_high_pct: float = 0.0  # 0 = disabled

    # --- Confidence-based position sizing tiers (1.0 = disabled) ---
    confidence_tier1_multiplier: float = 1.0
    confidence_tier2_multiplier: float = 1.0
    confidence_tier3_multiplier: float = 1.0
    confidence_tier4_multiplier: float = 1.0

    def confidence_multiplier(self, score: float) -> float:
        """Map 0–1 confidence score to a position-size multiplier."""
        if score >= 0.75:
            return self.confidence_tier4_multiplier
        if score >= 0.50:
            return self.confidence_tier3_multiplier
        if score >= 0.25:
            return self.confidence_tier2_multiplier
        return self.confidence_tier1_multiplier


def make_long_config() -> V4Config:
    """V4Config tuned for catalyst-driven momentum longs."""
    cfg = V4Config()
    cfg.stage1_min_price_change_pct = 0.15
    cfg.stage1_min_price = 2.00
    cfg.min_price = 2.00
    cfg.stage2_roc_min_pct = 0.07
    cfg.stage2_min_relative_volume = 10.0
    cfg.stage2_buying_pressure_min = 0.85
    cfg.max_position_pct = 0.50
    cfg.risk_per_trade = 0.02
    cfg.max_portfolio_heat = 0.12
    cfg.confidence_tier1_multiplier = 1.0
    cfg.confidence_tier2_multiplier = 2.0
    cfg.confidence_tier3_multiplier = 4.0
    cfg.confidence_tier4_multiplier = 8.0
    return cfg


def make_short_config() -> V4Config:
    """V4Config tuned for fade/short entries."""
    cfg = V4Config()
    return cfg
