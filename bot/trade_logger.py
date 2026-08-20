from __future__ import annotations
import csv
from pathlib import Path
from typing import List

from bot.intraday.types import TradeRecord

_FIELDS: List[str] = [
    "ticker", "direction", "entry_time", "entry_price", "shares",
    "stop_price", "target_price", "exit_time", "exit_price",
    "pnl", "exit_reason", "portfolio_heat_at_entry", "signals",
    "entry_slippage_pct", "exit_slippage_pct",
]


class TradeLogger:
    """Appends closed trades to a per-day CSV in log_dir."""

    def __init__(self, log_dir: str = "logs") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, record: TradeRecord) -> None:
        date_str = record.entry_time.strftime("%Y-%m-%d")
        path = self._log_dir / f"trades_{date_str}.csv"
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(self._to_row(record))

    def _to_row(self, r: TradeRecord) -> dict:
        def fmt_pct(v: float | None) -> str:
            return f"{v * 100:.3f}%" if v is not None else ""
        return {
            "ticker": r.ticker,
            "direction": r.direction,
            "entry_time": r.entry_time.isoformat(),
            "entry_price": r.entry_price,
            "shares": r.shares,
            "stop_price": r.stop_price,
            "target_price": r.target_price,
            "exit_time": r.exit_time.isoformat() if r.exit_time else "",
            "exit_price": r.exit_price if r.exit_price is not None else "",
            "pnl": round(r.pnl, 2) if r.pnl is not None else "",
            "exit_reason": r.exit_reason or "",
            "portfolio_heat_at_entry": round(r.portfolio_heat_at_entry, 4),
            "signals": "|".join(r.signals),
            "entry_slippage_pct": fmt_pct(r.entry_slippage_pct),
            "exit_slippage_pct": fmt_pct(r.actual_slippage_pct),
        }
