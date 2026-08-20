from __future__ import annotations
import csv
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

_FIELDS = ["date", "time_et", "ticker", "run_pct", "day_open", "hod_price", "etb_at_qualification"]


class ShortQualLogger:
    """Appends HOD-rejection short qualification events to a per-day CSV in log_dir.

    Measures how often qualifying candidates are actually shortable (ETB) at the
    moment they qualify, independent of the scrolling bot.log — see 2026-07-01
    finding that ETB availability, not the run% threshold, is the binding
    constraint on the short strategy's Alpaca-only execution path.
    """

    def __init__(self, log_dir: str = "logs") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        ticker: str,
        qualified_at: datetime,
        run_pct: float,
        day_open: float,
        hod_price: float,
        etb_at_qualification: bool,
    ) -> None:
        # bar.timestamp arrives UTC-aware (see BarStream.on_update) — convert to ET
        # for the log; naive datetimes are assumed UTC to match that source.
        ts_utc = qualified_at if qualified_at.tzinfo else qualified_at.replace(tzinfo=timezone.utc)
        ts_et = ts_utc.astimezone(_ET)
        date_str = ts_et.strftime("%Y-%m-%d")
        path = self._log_dir / f"short_quals_{date_str}.csv"
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "date": date_str,
                "time_et": ts_et.strftime("%H:%M:%S"),
                "ticker": ticker,
                "run_pct": round(run_pct, 2),
                "day_open": day_open,
                "hod_price": hod_price,
                "etb_at_qualification": etb_at_qualification,
            })
