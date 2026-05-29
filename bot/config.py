from __future__ import annotations
from dataclasses import dataclass
from bot.intraday.config import IntradayConfig


@dataclass
class V4Config(IntradayConfig):
    # --- V4 overrides of V3 defaults ---
    max_open_positions: int = 15
    max_sector_positions: int = 5
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
