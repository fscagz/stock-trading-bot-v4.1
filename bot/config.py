from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
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

    # --- Volume acceleration filter ---
    # Entry bar volume must be >= this fraction of the previous bar's volume.
    # Filters entries where the momentum spike has already peaked (volume declining).
    # 0.0 = disabled.
    stage2_min_vol_vs_prev_bar: float = 0.0

    # --- Break-even stop ---
    # Once unrealised profit reaches this multiple of ATR, raise stop to entry price.
    # 0.0 = disabled (never move stop to break-even).
    breakeven_trigger_atr_multiple: float = 0.0

    # --- Confidence-based position sizing tiers (1.0 = disabled) ---
    confidence_tier1_multiplier: float = 1.0
    confidence_tier2_multiplier: float = 1.0
    confidence_tier3_multiplier: float = 1.0
    confidence_tier4_multiplier: float = 1.0

    # --- Confidence score normalization ranges ---
    # roc_score  hits 1.0 at (1 + roc_range_mult)  × stage2_roc_min_pct
    # vol_score  hits 1.0 at (1 + vol_range_mult)  × stage2_min_relative_volume
    # Wider ranges spread catalyst plays across tiers instead of saturating at tier4.
    confidence_score_roc_range_mult: float = 3.0  # default matches legacy behaviour
    confidence_score_vol_range_mult: float = 3.0  # default matches legacy behaviour

    # --- Short-side entry filters (ignored when used as a long config) ---
    # Close must be in the bottom fraction of the bar's range (selling pressure).
    # 1.0 = disabled (no filter).
    short_selling_pressure_max: float = 1.0
    # Require the entry bar to be a red bar (close < open).
    short_require_red_bar: bool = False
    # Volume must be < prev_bar.volume * this ratio.  0.0 = disabled.
    short_volume_exhaustion_ratio: float = 0.0

    def confidence_multiplier(self, score: float) -> float:
        """Map 0–1 confidence score to a position-size multiplier."""
        if score >= 0.75:
            return self.confidence_tier4_multiplier
        if score >= 0.50:
            return self.confidence_tier3_multiplier
        if score >= 0.25:
            return self.confidence_tier2_multiplier
        return self.confidence_tier1_multiplier


@dataclass
class CombinedConfig:
    """Bundles a long and an optional short V4Config for use in the Simulator.

    When short is None the simulator runs long-only (identical to passing just
    a V4Config directly).  When short is set, the simulator runs both books
    with independent PortfolioState instances.
    """
    long: V4Config
    short: Optional[V4Config] = None


def make_standard_config() -> V4Config:
    """Standard config — original H baseline (rv=10, no gap-hold).

    Catalyst-driven momentum longs, $10M DV, 10× relative volume threshold.
    Use as a reference baseline or to run experiments against make_gap_hold_config().
    """
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
    cfg.confidence_tier3_multiplier = 3.0
    cfg.confidence_tier4_multiplier = 4.0
    # vol_score = 1.0 at 80× rel-vol (8× the 10× min); roc_score = 1.0 at 35% ROC (5× the 7% min)
    cfg.confidence_score_roc_range_mult = 4.0
    cfg.confidence_score_vol_range_mult = 7.0
    cfg.min_avg_dollar_volume = 10_000_000
    return cfg


def make_gap_hold_config() -> V4Config:
    """Gap-hold config — live paper trading configuration (rv=6 + gap-hold ≥5%).

    Catalyst-driven momentum longs + gap-and-hold entry signal.
    Relative volume threshold lowered to 6× to increase trade count.
    Gap-hold signal enabled in runner via _GAP_HOLD_* constants.
    """
    cfg = V4Config()
    cfg.stage1_min_price_change_pct = 0.15
    cfg.stage1_min_price = 2.00
    cfg.min_price = 2.00
    cfg.stage2_roc_min_pct = 0.07
    cfg.stage2_min_relative_volume = 6.0
    cfg.stage2_buying_pressure_min = 0.85
    cfg.max_position_pct = 0.50
    cfg.risk_per_trade = 0.02
    cfg.max_portfolio_heat = 0.12
    cfg.confidence_tier1_multiplier = 1.0
    cfg.confidence_tier2_multiplier = 2.0
    cfg.confidence_tier3_multiplier = 3.0
    cfg.confidence_tier4_multiplier = 4.0
    # vol_score = 1.0 at 48× rel-vol (8× the 6× min); roc_score = 1.0 at 35% ROC (5× the 7% min)
    cfg.confidence_score_roc_range_mult = 4.0
    cfg.confidence_score_vol_range_mult = 7.0
    cfg.min_avg_dollar_volume = 10_000_000
    return cfg


def make_short_config(**kwargs) -> V4Config:
    """V4Config tuned for intraday fade/short entries.

    Uses the same scanner universe as longs (high-momentum movers) but enters
    on exhaustion signals rather than continuation.  Conservative sizing by
    default (half long risk) since borrow availability is unverifiable historically.

    Pass keyword overrides to vary the entry filters in experiments:
        make_short_config(short_require_red_bar=True)
        make_short_config(short_volume_exhaustion_ratio=0.85)
    """
    cfg = V4Config()
    cfg.stage1_min_price_change_pct = 0.15
    cfg.stage1_min_price = 2.00
    cfg.min_price = 2.00
    cfg.stage2_roc_min_pct = 0.07
    cfg.stage2_min_relative_volume = 6.0
    cfg.risk_per_trade = 0.01          # conservative — half of long baseline
    cfg.max_portfolio_heat = 0.06
    cfg.max_position_pct = 0.25
    cfg.min_avg_dollar_volume = 10_000_000
    cfg.short_selling_pressure_max = 0.50  # close in bottom half of bar range
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def make_long_and_short_config(
    long_cfg: Optional[V4Config] = None,
    **short_kwargs,
) -> CombinedConfig:
    """Return a CombinedConfig using gap-hold longs (default) and tuned shorts.

    Pass long_cfg=make_standard_config() to use the standard long baseline instead.
    Pass short_kwargs to override individual short entry filters.
    """
    return CombinedConfig(
        long=long_cfg if long_cfg is not None else make_gap_hold_config(),
        short=make_short_config(**short_kwargs),
    )
