"""
Transaction Cost Creep Monitoring.

Tracks realized trade costs vs backtest assumptions and alerts
when costs exceed expected levels.

Usage
-----
    from monitoring.costs import CostCreepMonitor, CostCreepConfig

    costs_monitor = CostCreepMonitor(config)

    # After each trade execution
    costs_monitor.record_trade(
        ticker="AAPL",
        trade_date=date.today(),
        side="buy",
        notional=10000.0,
        expected_cost=50.0,      # from cost model at order time
        realized_cost=85.0,      # actual fill cost
    )

    alert, msg = costs_monitor.check_alert()
    if alert:
        print(msg)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CostCreepConfig:
    """Configuration for cost creep monitoring."""
    lookback_trades: int = 50      # number of recent trades to compare
    creep_alert_ratio: float = 2.0 # alert if realized/expected > this factor
    min_trades: int = 10           # minimum trades before alerting


DEFAULT_COST_CREEP_CONFIG = CostCreepConfig()


# ---------------------------------------------------------------------------
# Trade Record and Monitor
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Record of a single executed trade with cost tracking."""
    ticker: str
    trade_date: date
    side: str                  # 'buy' | 'sell'
    notional: float
    expected_cost: float       # from cost model at order time
    realized_cost: float       # actual cost (fill vs arrival price)

    @property
    def cost_ratio(self) -> float:
        """Ratio of realized to expected cost; NaN if expected == 0."""
        if self.expected_cost == 0:
            return float("nan")
        return self.realized_cost / self.expected_cost


class CostCreepMonitor:
    """
    Tracks realized trade costs vs backtest assumptions.

    Records each trade's expected and realized cost, computes rolling
    cost ratio, and alerts when ratio exceeds threshold.
    """

    def __init__(self, config: CostCreepConfig = DEFAULT_COST_CREEP_CONFIG):
        self.config = config
        self._records: List[TradeRecord] = []

    def record_trade(
        self,
        ticker: str,
        trade_date: date,
        side: str,
        notional: float,
        expected_cost: float,
        realized_cost: float,
    ) -> None:
        """Record a single executed trade."""
        record = TradeRecord(
            ticker=ticker,
            trade_date=trade_date,
            side=side,
            notional=notional,
            expected_cost=expected_cost,
            realized_cost=realized_cost,
        )
        self._records.append(record)

    def rolling_cost_ratio(
        self,
        n: Optional[int] = None,
    ) -> float:
        """
        Compute rolling mean cost ratio (realized / expected).

        Parameters
        ----------
        n : int, optional
            Number of recent trades to include. Defaults to config.lookback_trades.

        Returns
        -------
        float — mean realized/expected cost over last n trades; NaN if insufficient data
        """
        if len(self._records) == 0:
            return float("nan")

        lookback = n if n is not None else self.config.lookback_trades
        recent_trades = self._records[-lookback:]

        ratios = [t.cost_ratio for t in recent_trades]
        # Filter out NaN values
        ratios = [r for r in ratios if not np.isnan(r)]

        if len(ratios) == 0:
            return float("nan")

        return float(np.mean(ratios))

    def check_alert(self) -> Tuple[bool, str]:
        """
        Check if cost creep has exceeded alert threshold.

        Returns
        -------
        Tuple[bool, str] — (alert_triggered, message)
        """
        if len(self._records) < self.config.min_trades:
            return False, f"Insufficient trade history ({len(self._records)} trades recorded)."

        ratio = self.rolling_cost_ratio()

        if np.isnan(ratio):
            return False, "Unable to compute cost ratio (no valid trades)."

        if ratio > self.config.creep_alert_ratio:
            return (
                True,
                f"COST CREEP ALERT: Recent realized/expected cost ratio {ratio:.2f} "
                f"exceeds threshold {self.config.creep_alert_ratio}",
            )

        return False, f"Cost ratio {ratio:.2f} within normal range."

    def history_df(self) -> pd.DataFrame:
        """
        Return all trade records as DataFrame.

        Returns
        -------
        pd.DataFrame — columns: ticker, trade_date, side, notional, expected_cost,
                       realized_cost, cost_ratio
        """
        if not self._records:
            return pd.DataFrame()

        data = {
            "ticker": [r.ticker for r in self._records],
            "trade_date": [r.trade_date for r in self._records],
            "side": [r.side for r in self._records],
            "notional": [r.notional for r in self._records],
            "expected_cost": [r.expected_cost for r in self._records],
            "realized_cost": [r.realized_cost for r in self._records],
            "cost_ratio": [r.cost_ratio for r in self._records],
        }

        return pd.DataFrame(data)

    def total_cost_drag(self) -> float:
        """
        Compute total excess cost (realized - expected) across all trades.

        Returns
        -------
        float — cumulative extra cost paid vs backtest assumptions
        """
        if not self._records:
            return 0.0

        total_realized = sum(r.realized_cost for r in self._records)
        total_expected = sum(r.expected_cost for r in self._records)

        return total_realized - total_expected

    def summary(self) -> str:
        """Return formatted summary of cost creep status."""
        if not self._records:
            return "No trade history recorded.\n"

        alert, msg = self.check_alert()
        ratio = self.rolling_cost_ratio()
        total_drag = self.total_cost_drag()

        lines = [
            "",
            "=" * 70,
            "COST CREEP MONITOR SUMMARY",
            "=" * 70,
            f"Total Trades Recorded: {len(self._records)}",
            f"Recent Cost Ratio (realized/expected): {ratio:.2f}" if not np.isnan(ratio) else "N/A",
            f"Alert Threshold: {self.config.creep_alert_ratio}",
            f"Alert Status: {'YES' if alert else 'NO'}",
            "",
            f"Total Cost Drag: ${total_drag:,.2f}",
            f"  (realized costs exceed backtest assumptions by this amount)",
            "",
            f"Message: {msg}",
            "=" * 70,
            "",
        ]

        return "\n".join(lines)
