from __future__ import annotations
import logging
from typing import Optional, Tuple

from bot.intraday.config import IntradayConfig

logger = logging.getLogger(__name__)


class BracketManager:
    """Submits stop and limit-target exit orders after a confirmed entry fill."""

    def __init__(self, broker, config: IntradayConfig) -> None:
        self._broker = broker
        self._cfg = config

    def submit_bracket(
        self,
        ticker: str,
        side: str,
        shares: int,
        stop_price: float,
        target_price: float,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Submit stop and limit-target exits. Returns (stop_order_id, target_order_id)."""
        exit_side = "sell" if side == "long" else "buy"
        stop_id = None
        target_id = None

        try:
            stop_order = self._broker.submit_order(
                symbol=ticker,
                qty=shares,
                side=exit_side,
                type="stop",
                stop_price=round(stop_price, 2),
                time_in_force="day",
            )
            stop_id = stop_order.id
        except Exception as exc:
            logger.error("Stop order failed for %s: %s", ticker, exc)

        try:
            target_order = self._broker.submit_order(
                symbol=ticker,
                qty=shares,
                side=exit_side,
                type="limit",
                limit_price=round(target_price, 2),
                time_in_force="day",
            )
            target_id = target_order.id
        except Exception as exc:
            logger.error("Target order failed for %s: %s", ticker, exc)

        return stop_id, target_id
