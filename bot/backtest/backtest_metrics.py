from __future__ import annotations
from typing import List

from bot.intraday.types import TradeRecord


def compute_metrics(trades: List[TradeRecord], initial_equity: float) -> dict:
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl_per_trade": 0.0,
            "avg_winner": 0.0,
            "avg_loser": 0.0,
            "max_drawdown": 0.0,
            "avg_hold_minutes": 0.0,
            "exit_reasons": {},
        }

    pnls = [t.pnl for t in trades if t.pnl is not None]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    equity = initial_equity
    peak = equity
    max_drawdown = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    hold_minutes = []
    for t in trades:
        if t.exit_time and t.entry_time:
            hold_minutes.append((t.exit_time - t.entry_time).total_seconds() / 60)

    exit_reasons: dict = {}
    for t in trades:
        if t.exit_reason:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return {
        "total_trades": len(trades),
        "win_rate": len(winners) / len(pnls) if pnls else 0.0,
        "total_pnl": round(sum(pnls), 2),
        "avg_pnl_per_trade": round(sum(pnls) / len(pnls), 2) if pnls else 0.0,
        "avg_winner": round(sum(winners) / len(winners), 2) if winners else 0.0,
        "avg_loser": round(sum(losers) / len(losers), 2) if losers else 0.0,
        "max_drawdown": round(max_drawdown, 2),
        "avg_hold_minutes": round(sum(hold_minutes) / len(hold_minutes), 1) if hold_minutes else 0.0,
        "exit_reasons": exit_reasons,
    }
