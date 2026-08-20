from __future__ import annotations
from dataclasses import dataclass

from bot.intraday.config import IntradayConfig


@dataclass
class SizeResult:
    shares: float
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
    bar_volume: float = 0.0,
) -> SizeResult:
    """
    Compute share count using ATR-based risk sizing.

    shares = floor((equity × risk_per_trade) / (stop_atr_multiple × ATR))
    Capped at: floor((equity × max_position_pct) / entry_price)
    Optionally also capped at: bar_volume × max_bar_participation_pct — on a thin
    entry bar this can bind before the equity-based caps do, which is the point:
    a share count the entry bar's own volume can't absorb gets filled with the
    kind of market impact that shows up as slippage eating the stop cushion.
    """
    stop_distance = config.stop_atr_multiple * atr
    target_distance = config.target_atr_multiple * atr

    risk_dollars = equity * config.risk_per_trade
    uncapped_shares = round(risk_dollars / stop_distance, 3) if stop_distance > 0 else 0.0

    max_shares = round((equity * config.max_position_pct) / entry_price, 3) if entry_price > 0 else 0.0
    shares = min(uncapped_shares, max_shares)

    if config.max_bar_participation_pct > 0 and bar_volume > 0:
        participation_cap = round(bar_volume * config.max_bar_participation_pct, 3)
        shares = min(shares, participation_cap)

    capped = shares < uncapped_shares

    return SizeResult(
        shares=max(0.0, shares),
        stop_distance=stop_distance,
        target_distance=target_distance,
        capped=capped,
    )
