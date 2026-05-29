from __future__ import annotations
import logging
import time
from typing import Literal, Optional

from bot.intraday.config import IntradayConfig
from bot.intraday.execution.fill_tracker import FillResult, FillTracker

logger = logging.getLogger(__name__)
Side = Literal["long", "short"]


class OrderManager:
    """Submits aggressive limit entry orders via the Alpaca broker client."""

    def __init__(self, broker, config: IntradayConfig) -> None:
        self._broker = broker
        self._cfg = config
        self._fill_tracker = FillTracker(config)

    def limit_price(self, side: Side, current_price: float) -> float:
        """Return the aggressive limit price for the given side."""
        offset = self._cfg.limit_offset_pct
        return current_price * (1 + offset) if side == "long" else current_price * (1 - offset)

    def submit_entry(
        self,
        ticker: str,
        side: Side,
        shares: int,
        current_price: float,
    ) -> Optional[FillResult]:
        """Submit aggressive limit order. Poll until filled or timeout."""
        lmt = round(self.limit_price(side, current_price), 2)
        alpaca_side = "buy" if side == "long" else "sell"

        try:
            order = self._broker.submit_order(
                symbol=ticker,
                qty=shares,
                side=alpaca_side,
                type="limit",
                limit_price=lmt,
                time_in_force="day",
            )
        except Exception as exc:
            logger.error("Order submission failed for %s: %s", ticker, exc)
            return None

        deadline = time.time() + self._cfg.fill_timeout_seconds
        while time.time() < deadline:
            try:
                status = self._broker.get_order(order.id)
                if status.status in ("filled", "partially_filled"):
                    filled = int(status.filled_qty)
                    fill_price = float(status.filled_avg_price or lmt)
                    result = self._fill_tracker.process_fill(shares, filled, fill_price, lmt)
                    if result.should_cancel_remainder:
                        try:
                            self._broker.cancel_order(order.id)
                        except Exception:
                            pass
                    return result
                if status.status in ("canceled", "expired", "rejected"):
                    return None
            except Exception as exc:
                logger.warning("Fill poll failed for order %s: %s", order.id, exc)
            time.sleep(1)

        try:
            self._broker.cancel_order(order.id)
        except Exception:
            pass
        logger.warning("Order %s timed out for %s", order.id, ticker)
        return None
