"""Entry point for the V4 momentum bot.

Usage:
    python -m bot.main
    python -m bot.main --risk-scale 0.5
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

import bot.broker_alpaca as broker
from bot.config import V4Config, make_gap_hold_config, make_short_config
from bot.dashboard.state import DashboardState
from bot.live.runner import LiveRunner, _SHORT_HOD_MIN_RUN_PCT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 Momentum Bot")
    parser.add_argument(
        "--risk-scale", type=float, default=1.0,
        help="Scale risk_per_trade and max_portfolio_heat (default 1.0)",
    )
    args = parser.parse_args()

    api_key = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]

    ibkr_host = os.getenv("IBKR_HOST", "127.0.0.1")
    ibkr_port = int(os.getenv("IBKR_PORT", "4001"))
    ibkr_client_id = int(os.getenv("IBKR_CLIENT_ID_STREAM", "1"))
    ibkr_scanner_client_id = int(os.getenv("IBKR_CLIENT_ID_SCANNER", "3"))

    account = broker.get_account_info()
    equity = account["portfolio_value"]
    logger.info("Account equity: $%.2f | Status: %s | Paper: %s",
                equity, account["status"], broker._is_paper)

    long_config = make_gap_hold_config()
    # GapHold config at $10M — rv=6, bp=0.85, gap-hold ≥5% enabled.
    # Validated across 2022-2026: positive in all 5 periods, +35.7% in 2023, +79.6% in 2024.
    # Gap-hold signal parameters live in runner.py (_GAP_HOLD_* constants).
    long_config.target_atr_multiple = 4.0           # tier-4 chase entries use 4×ATR target
    long_config.stage2_min_dist_from_day_high_pct = 0.05  # skip entries within 5% of day high
    long_config.stage2_min_vol_vs_prev_bar = 0.80   # entry bar must have ≥80% of prev bar volume
    long_config.risk_per_trade = 0.010              # 2026-06-24: reduced to min (0.010) — 37 trades at -$196 avg, bypass raised to 12%
    long_config.stage2_min_relative_volume = 3.0   # experiment 2026-06-24: floor value (was 4.0) — maximize stage2 entries
    long_config.stage2_buying_pressure_min = 0.65  # experiment 2026-06-24: floor value (was 0.70) — maximize stage2 entries
    long_config.cooldown_enabled = False            # disabled for data gathering — re-enable when ready
    long_config.max_daily_drawdown = 0.05          # paper trading: raised from 0.02 — allow more experimentation
    long_config.breakeven_trigger_atr_multiple = 1.0  # 2026-06-25: lock stop at entry once up 1×ATR
    long_config.max_open_positions = 3             # cap concurrent positions; prevents batch blowup
    long_config.max_gap_hold_entries_per_day = 0  # 2026-06-26: removed cap — gathering data on trade quality (0 = unlimited)
    long_config.stop_atr_multiple = 1.6            # 2026-07-13: 1.5→1.6 — 2/3 exits today were hard_stop (66.7% > 65% rule)
    long_config.max_bar_participation_pct = 0.20    # 2026-07-14: cap entry size at 20% of entry-bar volume — 64 hard_stop
                                                     # trades across all history: median hold to stop was ~96s and combined
                                                     # entry+exit slippage (~0.55%) ate ~72% of the median 0.76% planned stop
                                                     # distance. Oversized fills relative to entry-bar liquidity on thin
                                                     # low-float gappers are the likely driver — this caps it directly.

    dash = DashboardState()
    dash.equity = equity
    dash.cash = account["cash"]
    dash.buying_power = account["buying_power"]
    dash.is_paper = broker._is_paper
    dash.config_snapshot = {
        "risk_per_trade": long_config.risk_per_trade,
        "max_portfolio_heat": long_config.max_portfolio_heat,
        "max_open_positions": long_config.max_open_positions,
        "stage1_min_price_change_pct": long_config.stage1_min_price_change_pct,
        "stage2_min_relative_volume": long_config.stage2_min_relative_volume,
        "stage2_buying_pressure_min": long_config.stage2_buying_pressure_min,
        "min_avg_dollar_volume": long_config.min_avg_dollar_volume,
        "confidence_tiers": (
            f"{long_config.confidence_tier1_multiplier:.0f}×"
            f"/{long_config.confidence_tier2_multiplier:.0f}×"
            f"/{long_config.confidence_tier3_multiplier:.0f}×"
            f"/{long_config.confidence_tier4_multiplier:.0f}×"
        ),
        "eod_evaluation": long_config.eod_evaluation,
        "risk_scale": args.risk_scale,
    }

    short_config = make_short_config()
    # HOD-rejection short: fade parabolics that ran ≥_SHORT_HOD_MIN_RUN_PCT% from open
    # after a 10-bar top. Only fires when SPY is below its 50-day MA (regime gate
    # computed in runner). risk_per_trade=1%, max_heat=6%, max_position=25%,
    # min_dollar_vol=$10M.

    dash.long_strategy_name = "gap_hold"
    dash.short_enabled = True
    dash.short_config_snapshot = {
        "strategy": "HOD Rejection",
        "min_run_pct": f"{_SHORT_HOD_MIN_RUN_PCT:.0f}%",
        "rejection_bars": "10 bars",
        "stop_target": "2× / 2× ATR",
        "regime_filter": "SPY < 50-day MA",
        "risk_per_trade": short_config.risk_per_trade,
        "max_portfolio_heat": short_config.max_portfolio_heat,
        "min_dollar_vol": short_config.min_avg_dollar_volume,
    }

    etb_set = broker.get_etb_set()
    logger.info("ETB set loaded: %d shortable symbols", len(etb_set))

    runner = LiveRunner(
        ibkr_host=ibkr_host,
        ibkr_port=ibkr_port,
        ibkr_client_id=ibkr_client_id,
        ibkr_scanner_client_id=ibkr_scanner_client_id,
        api_key=api_key,
        secret_key=secret_key,
        short_config=short_config,
        long_config=long_config,
        equity=equity,
        etb_set=etb_set,
        risk_scale=args.risk_scale,
        dash=dash,
        enable_shorts=True,
    )
    runner.run()


if __name__ == "__main__":
    main()
