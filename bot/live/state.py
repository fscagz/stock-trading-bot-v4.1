"""
Persists live session position metadata to disk across process restarts.

Written to bot/live/session_state.json on every entry, stop update, and exit.
On restart, the runner reads this file and reconciles with Alpaca's actual
open positions so no trade is lost or orphaned.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from bot.intraday.types import Position

_STATE_PATH = Path(__file__).parent / "session_state.json"
logger = logging.getLogger(__name__)


class SessionState:
    """Read/write position state to a JSON file.

    The file is keyed by today's date so a stale file from a previous session
    is automatically ignored on the next day's startup.
    """

    def __init__(self, path: Path = _STATE_PATH) -> None:
        self._path = path
        self._data: dict = {"date": None, "positions": {}}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with self._path.open() as f:
                self._data = json.load(f)
            logger.debug("SessionState loaded from %s", self._path)
        except Exception as exc:
            logger.warning("Could not load session state (%s): %s — starting fresh", self._path, exc)
            self._data = {"date": None, "positions": {}}

    def _save(self) -> None:
        try:
            with self._path.open("w") as f:
                json.dump(self._data, f, indent=2, default=str)
        except Exception as exc:
            logger.error("Could not write session state to %s: %s", self._path, exc)

    def _ensure_today(self) -> None:
        today = date.today().isoformat()
        if self._data.get("date") != today:
            self._data = {"date": today, "positions": {}}

    @property
    def is_today(self) -> bool:
        return self._data.get("date") == date.today().isoformat()

    def get_saved_positions(self) -> dict:
        """Return saved positions for today, or empty dict if the file is from a previous day."""
        if not self.is_today:
            return {}
        return dict(self._data.get("positions", {}))

    def save_position(self, position: Position) -> None:
        """Persist a newly opened position. Call immediately after the broker order succeeds."""
        self._ensure_today()
        self._data["positions"][position.ticker] = {
            "ticker": position.ticker,
            "direction": position.direction,
            "shares": position.shares,
            "entry_price": position.entry_price,
            "stop_price": position.stop_price,
            "target_price": position.target_price,
            "entry_time": position.entry_time.isoformat(),
            "atr_at_entry": position.atr_at_entry,
            "signals": position.signals,
            "sector": position.sector,
            "highest_close": position.highest_close,
            "entry_bar_volume": position.entry_bar_volume,
            "stop_order_id": str(position.stop_order_id) if position.stop_order_id else "",
        }
        self._save()
        logger.debug("SessionState: saved position %s", position.ticker)

    def update_stop(self, ticker: str, stop_price: float, stop_order_id: str) -> None:
        """Persist an updated stop price and the new broker stop order ID."""
        if not self.is_today:
            return
        pos = self._data.get("positions", {}).get(ticker)
        if pos is not None:
            pos["stop_price"] = stop_price
            pos["stop_order_id"] = stop_order_id
            self._save()

    def remove_position(self, ticker: str) -> None:
        """Remove a closed position. Call after the broker cover order succeeds."""
        if not self.is_today:
            return
        removed = self._data.get("positions", {}).pop(ticker, None)
        if removed is not None:
            self._save()
            logger.debug("SessionState: removed position %s", ticker)

    def save_gap_losses(self, losses: dict, entered_today: set) -> None:
        """Persist same-day gap-hold loss counts and entered symbols for mid-day restart recovery."""
        self._ensure_today()
        self._data["gap_hold_losses"] = {k: int(v) for k, v in losses.items()}
        self._data["entered_today"] = sorted(entered_today)
        self._save()

    def get_gap_losses(self) -> "tuple[dict, set]":
        """Return persisted gap losses and entered_today if the saved file is from today."""
        if not self.is_today:
            return {}, set()
        losses = {k: int(v) for k, v in self._data.get("gap_hold_losses", {}).items()}
        entered = set(self._data.get("entered_today", []))
        return losses, entered

    def restore_position(self, ticker: str) -> Optional[Position]:
        """Reconstruct a Position from saved state. Returns None if not found or corrupt."""
        saved = self.get_saved_positions().get(ticker)
        if not saved:
            return None
        try:
            entry_time = datetime.fromisoformat(saved["entry_time"])
            pos = Position(
                ticker=saved["ticker"],
                direction=saved["direction"],
                shares=int(saved["shares"]),
                entry_price=float(saved["entry_price"]),
                stop_price=float(saved["stop_price"]),
                target_price=float(saved["target_price"]),
                entry_time=entry_time,
                atr_at_entry=float(saved["atr_at_entry"]),
                signals=list(saved.get("signals", [])),
                sector=saved.get("sector", "Unknown"),
                highest_close=float(saved.get("highest_close", saved["entry_price"])),
                entry_bar_volume=int(saved.get("entry_bar_volume", 0)),
            )
            pos.stop_order_id = saved.get("stop_order_id", "")
            return pos
        except Exception as exc:
            logger.warning("Could not restore position %s from saved state: %s", ticker, exc)
            return None
