from __future__ import annotations
from dataclasses import dataclass

from bot.intraday.config import IntradayConfig


@dataclass
class SizeResult:
    shares: int
    stop_distance: float    # dollars from entry to stop
    target_distance: float  # dollars from entry to target
    capped: bool            # True if shares were reduced by max_position cap

    def long_stop(self, entry_price: float) -> float:
        return entry_price - self.stop_distance

    def long_target(self, entry_price: float) -> float:
        return entry_price + self.target_distance

    def short_stop(self, entry_price: float) -> float:
        return entry_price + self.stop_distance

    def short_target(self, entry_price: float) -> float:
        return entry_price - self.target_distance


def compute_position_size(
    equity: float,
    atr: float,
    entry_price: float,
    config: IntradayConfig,
) -> SizeResult:
    """
    Compute share count using ATR-based risk sizing.

    shares = floor((equity × risk_per_trade) / (stop_atr_multiple × ATR))
    Capped at: floor((equity × max_position_pct) / entry_price)
    """
    stop_distance = config.stop_atr_multiple * atr
    target_distance = config.target_atr_multiple * atr

    risk_dollars = equity * config.risk_per_trade
    uncapped_shares = int(risk_dollars / stop_distance) if stop_distance > 0 else 0

    max_shares = int((equity * config.max_position_pct) / entry_price) if entry_price > 0 else 0
    capped = uncapped_shares > max_shares
    shares = min(uncapped_shares, max_shares)

    return SizeResult(
        shares=max(0, shares),
        stop_distance=stop_distance,
        target_distance=target_distance,
        capped=capped,
    )
