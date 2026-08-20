from dataclasses import dataclass


@dataclass
class IntradayConfig:
    # --- Risk sizing (all as fractions of equity) ---
    risk_per_trade: float = 0.005        # 0.5% of equity risked per trade
    max_position_pct: float = 0.05       # 5% of equity max position value
    stop_atr_multiple: float = 1.5       # stop = 1.5 × ATR(14) from entry
    target_atr_multiple: float = 3.0     # target = 3.0 × ATR(14) from entry (2:1 R/R)
    max_daily_drawdown: float = 0.02     # -2% of equity triggers kill switch
    max_portfolio_heat: float = 0.03     # total open risk ≤ 3% of equity
    max_open_positions: int = 5
    max_sector_positions: int = 2
    max_position_correlation: float = 0.7
    max_bar_participation_pct: float = 0.0  # 0 = disabled; else cap shares at this fraction of entry-bar volume

    # --- Execution ---
    limit_offset_pct: float = 0.0005     # bid+0.05% for longs; ask-0.05% for shorts
    fill_timeout_seconds: int = 30
    min_fill_ratio: float = 0.80         # cancel remainder if < 80% filled

    # --- Session windows (ET) ---
    session_start: str = "09:45"
    session_end: str = "15:30"
    eod_close: str = "15:45"

    # --- Cooldown ---
    cooldown_enabled: bool = True
    consecutive_loss_trigger: int = 3
    consecutive_loss_cooldown_minutes: int = 30

    # --- Technical signal thresholds ---
    vwap_deviation_pct: float = 0.015
    vwap_volume_ratio: float = 1.5
    momentum_volume_ratio: float = 2.0
    momentum_close_range_pct: float = 0.25
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    breakout_volume_ratio: float = 1.5

    # --- Event signal thresholds ---
    earnings_surprise_min_pct: float = 0.05
    earnings_gap_skip_pct: float = 0.04
    analyst_gap_skip_pct: float = 0.03
    analyst_target_raise_pct: float = 0.10
    opening_range_minutes: int = 15

    # --- Sentiment thresholds ---
    sentiment_threshold: float = 0.65
    sentiment_half_life_minutes: float = 30.0
    sentiment_price_move_skip_pct: float = 0.01

    # --- Universe filters ---
    min_avg_dollar_volume: float = 20_000_000.0
    min_atr_pct: float = 0.0075
    min_price: float = 5.0
    max_price: float = 500.0
    max_spread_pct: float = 0.001

    # --- Regime thresholds ---
    vix_high_vol: float = 30.0
    vix_crash: float = 45.0
    adx_trend_threshold: float = 25.0
    spy_gap_crash_pct: float = 0.03

    # --- ML (Phase 4) ---
    ml_min_probability: float = 0.55
    ml_size_tiers: tuple = (0.55, 0.65, 0.75)
