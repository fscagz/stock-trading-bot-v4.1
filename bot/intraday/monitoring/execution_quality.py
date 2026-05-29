"""
Execution quality monitor.

Tracks actual vs expected slippage and fill rates from the trade log.
Alert if actual/expected ratio > 2× over a rolling window.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class ExecutionReport:
    n_trades: int
    avg_actual_slippage_pct: float
    avg_expected_slippage_pct: float
    slippage_ratio: float           # actual / expected
    alert: bool                     # True if ratio >= kill_multiple


def compute_execution_quality(
    trade_log_path: str,
    window_days: int = 30,
    kill_multiple: float = 2.0,
) -> Optional[ExecutionReport]:
    df = pd.read_csv(trade_log_path, parse_dates=["entry_time"])
    df = df[df["exit_price"].notna()].copy()

    for col in ["actual_slippage_pct", "expected_slippage_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["actual_slippage_pct", "expected_slippage_pct"])
    if df.empty:
        return None

    cutoff = df["entry_time"].max() - pd.Timedelta(days=window_days)
    df = df[df["entry_time"] >= cutoff]
    if df.empty:
        return None

    avg_actual = float(df["actual_slippage_pct"].mean())
    avg_expected = float(df["expected_slippage_pct"].mean())
    ratio = avg_actual / avg_expected if avg_expected > 0 else 0.0

    return ExecutionReport(
        n_trades=len(df),
        avg_actual_slippage_pct=avg_actual,
        avg_expected_slippage_pct=avg_expected,
        slippage_ratio=ratio,
        alert=ratio >= kill_multiple,
    )
