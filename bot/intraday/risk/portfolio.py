from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from bot.intraday.config import IntradayConfig
from bot.intraday.types import Position


@dataclass
class PortfolioState:
    equity: float
    config: IntradayConfig
    positions: Dict[str, Position] = field(default_factory=dict)
    session_start_equity: float = 0.0
    consecutive_losses: int = 0
    cooldown_until: Optional[datetime] = None
    kill_switch_active: bool = False

    def __post_init__(self) -> None:
        if self.session_start_equity == 0.0:
            self.session_start_equity = self.equity

    def add_position(self, position: Position) -> None:
        self.positions[position.ticker] = position

    def remove_position(self, ticker: str) -> Optional[Position]:
        return self.positions.pop(ticker, None)

    @property
    def portfolio_heat_pct(self) -> float:
        """Total open risk as fraction of equity."""
        total_risk = sum(p.open_risk for p in self.positions.values())
        return total_risk / self.equity if self.equity > 0 else 0.0

    def exceeds_heat_limit(self) -> bool:
        return self.portfolio_heat_pct > self.config.max_portfolio_heat

    def sector_count(self, sector: str) -> int:
        return sum(1 for p in self.positions.values() if p.sector == sector)

    def exceeds_sector_cap(self, sector: str) -> bool:
        return self.sector_count(sector) >= self.config.max_sector_positions

    def at_max_positions(self) -> bool:
        return len(self.positions) >= self.config.max_open_positions

    def daily_pnl_pct(self) -> float:
        if self.session_start_equity == 0:
            return 0.0
        return (self.equity - self.session_start_equity) / self.session_start_equity

    def in_cooldown(self, now: datetime) -> bool:
        return self.cooldown_until is not None and now < self.cooldown_until

    def can_enter(self, sector: str, now: datetime) -> Tuple[bool, str]:
        """Return (allowed, reason). Reason is empty string if allowed."""
        if self.kill_switch_active:
            return False, "kill_switch_active"
        if self.in_cooldown(now):
            return False, "in_cooldown"
        if self.at_max_positions():
            return False, "max_positions_reached"
        if self.exceeds_heat_limit():
            return False, "portfolio_heat_exceeded"
        if self.exceeds_sector_cap(sector):
            return False, "sector_cap_exceeded"
        return True, ""
