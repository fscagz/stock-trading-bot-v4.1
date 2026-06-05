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
from bot.config import V4Config, make_long_config
from bot.dashboard.state import DashboardState
from bot.live.runner import LiveRunner

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

    long_config = make_long_config()

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
        "risk_scale": args.risk_scale,
    }

    # Short strategy disabled — pass empty ETB set so short entries never fire.
    runner = LiveRunner(
        ibkr_host=ibkr_host,
        ibkr_port=ibkr_port,
        ibkr_client_id=ibkr_client_id,
        ibkr_scanner_client_id=ibkr_scanner_client_id,
        api_key=api_key,
        secret_key=secret_key,
        short_config=V4Config(),
        long_config=long_config,
        equity=equity,
        etb_set=set(),
        risk_scale=args.risk_scale,
        dash=dash,
    )
    runner.run()


if __name__ == "__main__":
    main()
