"""Entry point for the V4 live short-momentum runner.

Usage:
    python -m bot.live
    python -m bot.live --risk-scale 0.5
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

import bot.broker_alpaca as broker
from bot.config import V4Config, make_long_config
from bot.live.runner import LiveRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 Momentum Short Live Runner")
    parser.add_argument(
        "--risk-scale", type=float, default=1.0,
        help="Scale risk_per_trade, max_position_pct, max_portfolio_heat (default 1.0)",
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
    logger.info("Account equity: $%.2f | Status: %s", equity, account["status"])

    # Short strategy is deprecated — pass empty ETB set so short entries never fire.
    etb_set: set = set()

    runner = LiveRunner(
        ibkr_host=ibkr_host,
        ibkr_port=ibkr_port,
        ibkr_client_id=ibkr_client_id,
        ibkr_scanner_client_id=ibkr_scanner_client_id,
        api_key=api_key,
        secret_key=secret_key,
        short_config=V4Config(),
        long_config=make_long_config(),
        equity=equity,
        etb_set=etb_set,
        risk_scale=args.risk_scale,
    )
    runner.run()


if __name__ == "__main__":
    main()
