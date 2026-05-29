"""
Model decay monitor.

Compares live model accuracy (from trade log) against walk-forward baseline.
If accuracy drops > decay_threshold percentage points, triggers a retrain flag.
"""
from __future__ import annotations
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_live_accuracy(trade_log_path: str, window_days: int = 30) -> float:
    """
    Compute model accuracy over the last window_days of closed trades.
    Accuracy = fraction where ml_score >= 0.55 AND exit_reason == 'target'
               + fraction where ml_score < 0.55 AND exit_reason != 'target'.
    Trades without ml_score are excluded.
    """
    df = pd.read_csv(trade_log_path, parse_dates=["entry_time"])
    df = df[df["exit_price"].notna() & df["ml_score"].notna()].copy()
    if df.empty:
        return 0.0

    cutoff = df["entry_time"].max() - pd.Timedelta(days=window_days)
    df = df[df["entry_time"] >= cutoff]
    if df.empty:
        return 0.0

    df["ml_score"] = pd.to_numeric(df["ml_score"], errors="coerce")
    df["predicted_win"] = df["ml_score"] >= 0.55
    df["actual_win"] = df["exit_reason"] == "target"
    return float((df["predicted_win"] == df["actual_win"]).mean())


def check_decay(
    live_accuracy: float,
    baseline_accuracy: float,
    decay_threshold: float = 0.05,
) -> bool:
    """Return True if live accuracy has decayed beyond threshold (retrain needed)."""
    decayed = (baseline_accuracy - live_accuracy) >= decay_threshold
    if decayed:
        logger.warning(
            "Model decay detected: live=%.3f baseline=%.3f (delta=%.3f > threshold=%.3f)",
            live_accuracy, baseline_accuracy,
            baseline_accuracy - live_accuracy, decay_threshold,
        )
    return decayed
