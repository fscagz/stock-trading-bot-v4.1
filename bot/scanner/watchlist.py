from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Dict, Optional, Set

from bot.config import V4Config

if TYPE_CHECKING:
    from bot.intraday.data.stream import BarStream

logger = logging.getLogger(__name__)

_TRADING_MINUTES_PER_DAY = 390


class Watchlist:
    """Manages the set of candidate symbols and their BarStream subscriptions.

    When a symbol is added, its 20-day average per-minute volume baseline is loaded
    once and cached. This baseline is used by MomentumValidator for relative volume checks.
    """

    def __init__(self, stream: BarStream, config: V4Config) -> None:
        self._stream = stream
        self._cfg = config
        self._symbols: Set[str] = set()
        self._baselines: Dict[str, float] = {}

    @property
    def symbols(self) -> Set[str]:
        return self._symbols

    def add(self, symbol: str, high_priority: bool = False) -> None:
        if symbol in self._symbols:
            return
        baseline = self._load_baseline_volume(symbol)
        self._baselines[symbol] = baseline
        self._symbols.add(symbol)
        self._stream.subscribe(symbol)
        logger.info("Watchlist +%s (baseline_vol=%.0f/min, high_priority=%s)", symbol, baseline, high_priority)

    def remove(self, symbol: str) -> None:
        self._symbols.discard(symbol)
        self._baselines.pop(symbol, None)
        self._stream.unsubscribe(symbol)
        logger.info("Watchlist -%s", symbol)

    def get_baseline_volume(self, symbol: str) -> Optional[float]:
        return self._baselines.get(symbol)

    def _load_baseline_volume(self, symbol: str) -> float:
        try:
            from bot.data.daily_loader import get_daily
            df = get_daily(symbol, period="1mo")
            if df.empty or "volume" not in df.columns:
                return 0.0
            avg_daily_volume = df["volume"].tail(20).mean()
            return float(avg_daily_volume) / _TRADING_MINUTES_PER_DAY
        except Exception as exc:
            logger.warning("Could not load baseline volume for %s: %s", symbol, exc)
            return 0.0
