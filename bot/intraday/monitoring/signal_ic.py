"""
Rolling 30-day Information Coefficient (IC) per signal type.

IC = Spearman correlation between signal presence (0/1) and trade outcome (hit_target).
A signal with IC < 0 over 3 consecutive months should be flagged for review.
"""
from __future__ import annotations
from typing import Optional

import pandas as pd
from scipy.stats import spearmanr


def compute_signal_ic(
    trade_log_path: str,
    signal_keyword: str,
    window_days: int = 30,
) -> pd.DataFrame:
    """
    Compute rolling IC for a signal keyword (e.g. 'vwap_continuation').

    Returns DataFrame with columns: entry_time, ic, n_trades.
    """
    df = pd.read_csv(trade_log_path, parse_dates=["entry_time"])
    df = df[df["exit_price"].notna()].copy()
    df["hit_target"] = (df["exit_reason"] == "target").astype(int)
    df["has_signal"] = df["signals"].str.contains(signal_keyword, na=False).astype(int)
    df = df.sort_values("entry_time").reset_index(drop=True)

    records = []
    for i, row in df.iterrows():
        window_end = row["entry_time"]
        window_start = window_end - pd.Timedelta(days=window_days)
        window_df = df[(df["entry_time"] >= window_start) & (df["entry_time"] <= window_end)]
        if len(window_df) < 10:
            records.append({"entry_time": window_end, "ic": None, "n_trades": len(window_df)})
            continue
        ic, _ = spearmanr(window_df["has_signal"], window_df["hit_target"])
        records.append({"entry_time": window_end, "ic": ic, "n_trades": len(window_df)})

    return pd.DataFrame(records)
