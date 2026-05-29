from __future__ import annotations
from dataclasses import dataclass

from bot.intraday.config import IntradayConfig


@dataclass
class FillResult:
    filled_shares: int
    fill_ratio: float           # filled / ordered
    fill_price: float
    slippage_pct: float         # |fill_price - limit_price| / limit_price
    complete: bool              # True if fill_ratio >= min_fill_ratio
    should_cancel_remainder: bool


class FillTracker:
    """Evaluates fill quality and handles partial fill decisions."""

    def __init__(self, config: IntradayConfig) -> None:
        self._cfg = config

    def process_fill(
        self,
        ordered: int,
        filled: int,
        fill_price: float,
        limit_price: float,
    ) -> FillResult:
        fill_ratio = filled / ordered if ordered > 0 else 0.0
        slippage_pct = (
            abs(fill_price - limit_price) / limit_price if limit_price > 0 else 0.0
        )
        complete = fill_ratio >= self._cfg.min_fill_ratio
        return FillResult(
            filled_shares=filled,
            fill_ratio=fill_ratio,
            fill_price=fill_price,
            slippage_pct=slippage_pct,
            complete=complete,
            should_cancel_remainder=not complete and filled > 0,
        )
