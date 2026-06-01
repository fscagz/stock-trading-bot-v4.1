from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from bot.intraday.types import Position

logger = logging.getLogger(__name__)
_STATE_FILE = Path("positions_state.json")


def save(positions: Dict[str, Position], path: Path = _STATE_FILE) -> None:
    data = {
        sym: {
            "ticker": p.ticker,
            "direction": p.direction,
            "shares": p.shares,
            "entry_price": p.entry_price,
            "stop_price": p.stop_price,
            "target_price": p.target_price,
            "entry_time": p.entry_time.isoformat(),
            "atr_at_entry": p.atr_at_entry,
            "signals": p.signals,
            "sector": p.sector,
            "open_risk": p.open_risk,
            "highest_close": p.highest_close,
            "stop_order_id": p.stop_order_id,
            "entry_bar_volume": p.entry_bar_volume,
        }
        for sym, p in positions.items()
    }
    path.write_text(json.dumps(data, indent=2))


def load(path: Path = _STATE_FILE) -> Dict[str, Position]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Could not read position state file: %s", exc)
        return {}
    positions = {}
    for sym, d in data.items():
        positions[sym] = Position(
            ticker=d["ticker"],
            direction=d["direction"],
            shares=d["shares"],
            entry_price=d["entry_price"],
            stop_price=d["stop_price"],
            target_price=d["target_price"],
            entry_time=datetime.fromisoformat(d["entry_time"]),
            atr_at_entry=d["atr_at_entry"],
            signals=d["signals"],
            sector=d["sector"],
            open_risk=d["open_risk"],
            highest_close=d["highest_close"],
            stop_order_id=d.get("stop_order_id", ""),
            entry_bar_volume=d.get("entry_bar_volume", 0),
        )
    return positions


def sync_from_broker(portfolio, broker_client) -> None:
    """
    Reconcile saved position state with live Alpaca positions on bot startup.
    Restores known positions, warns about discrepancies.
    """
    saved = load()
    try:
        live = {p.symbol: p for p in broker_client.get_all_positions()}
    except Exception as exc:
        logger.error("Could not fetch broker positions on startup: %s", exc)
        return

    for sym, pos in saved.items():
        if sym in live:
            bp = live[sym]
            actual_shares = int(float(bp.qty))
            if actual_shares != pos.shares:
                logger.warning(
                    "%s: state has %d shares, broker has %d — using broker qty",
                    sym, pos.shares, actual_shares,
                )
                pos.shares = actual_shares
            portfolio.add_position(pos)
            logger.info(
                "Restored position: %s %s %d shares @ %.2f stop=%.2f target=%.2f",
                sym, pos.direction, pos.shares, pos.entry_price,
                pos.stop_price, pos.target_price,
            )
        else:
            logger.warning(
                "%s: in state file but not in broker — was closed externally, skipping",
                sym,
            )

    for sym in live:
        if sym not in saved:
            logger.warning(
                "%s: open in broker but not in state file — unknown position, manual review needed",
                sym,
            )
