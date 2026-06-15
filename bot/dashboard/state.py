from __future__ import annotations
import threading
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from bot.intraday.types import Position, TradeRecord


class DashboardState:
    def __init__(self) -> None:
        # Account snapshot (populated once at startup)
        self.equity: float = 0.0
        self.cash: float = 0.0
        self.buying_power: float = 0.0
        self.is_paper: bool = True

        # Portfolio (synced from PortfolioState on every position change)
        self.positions: Dict[str, Position] = {}
        self.portfolio_heat_pct: float = 0.0
        self.kill_switch_active: bool = False
        self.consecutive_losses: int = 0
        self.cooldown_until: Optional[datetime] = None

        # Regime (synced once per trading day)
        self.regime_uptrend: bool = True
        self.regime_date: Optional[date] = None

        # Trades closed this session
        self.closed_trades: List[TradeRecord] = []

        # Last bar close price per symbol (for unrealized P&L)
        self.last_prices: Dict[str, float] = {}

        # Config key-value snapshot (populated once at startup)
        self.config_snapshot: Dict[str, Any] = {}

        # Short strategy state
        self.short_enabled: bool = False
        self.short_heat_pct: float = 0.0
        self.short_allowed: bool = True        # SPY < 50-day MA (re-evaluated each day)
        self.short_config_snapshot: Dict[str, Any] = {}

        # Which long entry strategy is active
        self.long_strategy_name: str = "gap_hold"

        self._lock = threading.RLock()
