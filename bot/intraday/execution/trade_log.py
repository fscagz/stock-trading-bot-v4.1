from __future__ import annotations
import csv
import os
from datetime import datetime
from typing import List

from bot.intraday.types import TradeRecord

FIELDNAMES = [
    "ticker", "direction", "entry_time", "entry_price", "shares",
    "stop_price", "target_price", "signals", "sector", "regime",
    "portfolio_heat_at_entry", "expected_slippage_pct", "ml_score",
    "exit_time", "exit_price", "actual_slippage_pct", "pnl", "exit_reason",
]


class TradeLogger:
    """Appends trade entries to CSV and updates rows with exit data in-place."""

    def __init__(self, path: str) -> None:
        self._path = path
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    def log_entry(self, record: TradeRecord) -> None:
        with open(self._path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow({
                "ticker": record.ticker,
                "direction": record.direction,
                "entry_time": record.entry_time.isoformat(),
                "entry_price": record.entry_price,
                "shares": record.shares,
                "stop_price": record.stop_price,
                "target_price": record.target_price,
                "signals": "|".join(record.signals),
                "sector": record.sector,
                "regime": record.regime,
                "portfolio_heat_at_entry": record.portfolio_heat_at_entry,
                "expected_slippage_pct": record.expected_slippage_pct,
                "ml_score": record.ml_score if record.ml_score is not None else "",
                "exit_time": "",
                "exit_price": "",
                "actual_slippage_pct": "",
                "pnl": "",
                "exit_reason": "",
            })

    def log_exit(
        self,
        ticker: str,
        entry_time: datetime,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        actual_slippage_pct: float,
    ) -> None:
        """Find the open row for (ticker, entry_time) and fill in exit fields."""
        rows = self._read_all()
        entry_ts = entry_time.isoformat()
        for row in rows:
            if (row["ticker"] == ticker
                    and row["entry_time"] == entry_ts
                    and row["exit_price"] == ""):
                shares = int(row["shares"])
                entry_price = float(row["entry_price"])
                direction = row["direction"]
                if direction == "long":
                    pnl = (exit_price - entry_price) * shares
                else:
                    pnl = (entry_price - exit_price) * shares
                row["exit_time"] = exit_time.isoformat()
                row["exit_price"] = exit_price
                row["exit_reason"] = exit_reason
                row["actual_slippage_pct"] = actual_slippage_pct
                row["pnl"] = round(pnl, 4)
                break
        self._write_all(rows)

    def _read_all(self) -> List[dict]:
        with open(self._path, newline="") as f:
            return list(csv.DictReader(f))

    def _write_all(self, rows: List[dict]) -> None:
        with open(self._path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
