from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

Direction = Literal["long", "short"]


@dataclass
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3


@dataclass
class Signal:
    ticker: str
    direction: Direction
    signal_type: str
    timestamp: datetime
    bar: Bar
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    ticker: str
    direction: Direction
    shares: int
    entry_price: float
    stop_price: float
    target_price: float
    entry_time: datetime
    atr_at_entry: float
    signals: List[str]
    sector: str
    open_risk: float = 0.0

    def __post_init__(self) -> None:
        if self.open_risk == 0.0:
            self.open_risk = self.shares * abs(self.entry_price - self.stop_price)


@dataclass
class TradeRecord:
    ticker: str
    direction: Direction
    entry_time: datetime
    entry_price: float
    shares: int
    stop_price: float
    target_price: float
    signals: List[str]
    sector: str
    regime: str
    portfolio_heat_at_entry: float
    expected_slippage_pct: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    actual_slippage_pct: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None   # "stop" | "target" | "eod_close" | "kill_switch"
    ml_score: Optional[float] = None

    @property
    def is_closed(self) -> bool:
        return self.exit_price is not None
