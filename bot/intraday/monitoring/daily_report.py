"""
EOD daily report generator.

Prints a summary of the day's trades: P&L, win rate, breakdown by signal and regime,
and slippage quality. Run at end of session or next morning.

Usage:
    python -m bot.intraday.monitoring.daily_report
"""
from __future__ import annotations
from datetime import date

import pandas as pd


SIGNAL_KEYWORDS = [
    "vwap_continuation", "momentum_burst", "breakout",
    "rsi_extreme", "earnings", "sentiment", "ma_crossover", "analyst",
]


def generate_daily_report(
    trade_log_path: str,
    report_date: date = None,
) -> str:
    report_date = report_date or date.today()
    df = pd.read_csv(trade_log_path, parse_dates=["entry_time"])
    today = df[df["entry_time"].dt.date == report_date].copy()
    closed = today[today["exit_price"].notna()].copy()

    if closed.empty:
        return f"=== {report_date} — No closed trades ==="

    closed["pnl"] = pd.to_numeric(closed["pnl"], errors="coerce")
    total_pnl = closed["pnl"].sum()
    win_rate = (closed["exit_reason"] == "target").mean()
    trade_count = len(closed)

    lines = [
        f"=== Daily Report {report_date} ===",
        f"Trades: {trade_count}  |  Win Rate: {win_rate:.1%}  |  P&L: ${total_pnl:.2f}",
        "",
        "By signal:",
    ]

    for kw in SIGNAL_KEYWORDS:
        mask = closed["signals"].str.contains(kw, na=False)
        n = int(mask.sum())
        if n > 0:
            sig_pnl = closed.loc[mask, "pnl"].sum()
            lines.append(f"  {kw}: {n} trades  P&L=${sig_pnl:.2f}")

    lines += ["", "By regime:"]
    for regime, group in closed.groupby("regime"):
        g_pnl = group["pnl"].sum()
        lines.append(f"  {regime}: {len(group)} trades  P&L=${g_pnl:.2f}")

    closed["actual_slippage_pct"] = pd.to_numeric(closed["actual_slippage_pct"], errors="coerce")
    closed["expected_slippage_pct"] = pd.to_numeric(closed["expected_slippage_pct"], errors="coerce")
    avg_actual = closed["actual_slippage_pct"].mean()
    avg_expected = closed["expected_slippage_pct"].mean()

    if avg_expected and avg_expected > 0:
        ratio = avg_actual / avg_expected
        lines += [
            "",
            f"Slippage: actual={avg_actual:.4%}  expected={avg_expected:.4%}  ratio={ratio:.2f}x",
        ]
    else:
        lines += ["", "Slippage: N/A"]

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "bot/trade_log.csv"
    print(generate_daily_report(path))
