"""
Persistent cache for point-in-time fundamental data.

Stores fundamentals as Parquet files partitioned by data source.
Every record must include filing_date (the date the market could have
known the value) and period_end_date (the accounting period end).

Schema (required columns in every stored DataFrame)
----------------------------------------------------
ticker          : str
filing_date     : date  — SEC publish / EDGAR accession date ⚠️ point-in-time key
period_end_date : date  — fiscal period end
period_type     : str   — 'annual' | 'quarterly'
revenue         : float | NaN
net_income      : float | NaN
eps_diluted     : float | NaN
gross_profit    : float | NaN
ebitda          : float | NaN
total_assets    : float | NaN
total_equity    : float | NaN
total_debt      : float | NaN
free_cash_flow  : float | NaN

Derived metrics (pe_ratio, fcf_yield, etc.) are NOT stored here — they are
computed at query time in simfin_loader.get_fundamentals_as_of() because
they require a point-in-time price, which is loaded separately.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

# Required columns — every stored record must contain these.
REQUIRED_COLUMNS = [
    "ticker",
    "filing_date",
    "period_end_date",
    "period_type",
]

# Numeric line items stored as-is from the source.
NUMERIC_COLUMNS = [
    "revenue",
    "net_income",
    "eps_diluted",
    "gross_profit",
    "ebitda",
    "total_assets",
    "total_equity",
    "total_debt",
    "free_cash_flow",
]

ALL_COLUMNS = REQUIRED_COLUMNS + NUMERIC_COLUMNS


def _store_path(store_dir: Path, source: str) -> Path:
    return store_dir / f"fundamentals_{source}.parquet"


def save_fundamentals(
    df: pd.DataFrame,
    source: str,
    store_dir: Optional[Path] = None,
    merge: bool = True,
) -> Path:
    """
    Persist a fundamentals DataFrame to Parquet.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all REQUIRED_COLUMNS. Numeric columns are coerced to float.
    source : str
        Data source label, e.g. 'simfin' or 'edgar'. Used in the filename.
    store_dir : Path, optional
        Directory for Parquet files. Defaults to config.CACHE_DIR / 'fundamentals'.
    merge : bool
        If True and a file already exists, merge new data (newer filing_date wins
        for duplicate ticker+period_end_date+period_type rows). If False, overwrite.

    Returns
    -------
    Path
        Path to the written Parquet file.
    """
    if store_dir is None:
        from config import CACHE_DIR
        store_dir = CACHE_DIR / "fundamentals"
    store_dir.mkdir(parents=True, exist_ok=True)

    # Validate required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    df = df.copy()

    # Coerce date columns
    for col in ("filing_date", "period_end_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.date

    # Coerce numeric columns
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    path = _store_path(store_dir, source)

    if merge and path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
        # Keep the most recently filed record for each (ticker, period_end_date, period_type)
        combined = combined.sort_values("filing_date")
        combined = combined.drop_duplicates(
            subset=["ticker", "period_end_date", "period_type"], keep="last"
        )
        df = combined

    df.to_parquet(path, index=False)
    return path


def load_fundamentals(
    source: str,
    store_dir: Optional[Path] = None,
    tickers: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load all cached fundamental records for a source.

    Parameters
    ----------
    source : str
        Data source label ('simfin' or 'edgar').
    store_dir : Path, optional
        Directory for Parquet files. Defaults to config.CACHE_DIR / 'fundamentals'.
    tickers : list, optional
        If provided, filter to these tickers only.

    Returns
    -------
    pd.DataFrame
        All stored records. Empty DataFrame if no cache file exists.
    """
    if store_dir is None:
        from config import CACHE_DIR
        store_dir = CACHE_DIR / "fundamentals"

    path = _store_path(store_dir, source)
    if not path.exists():
        return pd.DataFrame(columns=ALL_COLUMNS)

    df = pd.read_parquet(path)
    if tickers is not None:
        df = df[df["ticker"].isin(tickers)]
    return df.reset_index(drop=True)


def get_fundamentals_as_of(
    ticker: str,
    as_of: date,
    source: str = "simfin",
    period_type: str = "annual",
    store_dir: Optional[Path] = None,
    max_staleness_days: int = 180,
) -> Optional[pd.Series]:
    """
    Return the most recent fundamental record for a ticker where
    filing_date <= as_of (point-in-time safe).

    Parameters
    ----------
    ticker : str
    as_of : date
        Rebalance date. Only records filed on or before this date are considered.
    source : str
        Data source to query ('simfin' or 'edgar').
    period_type : str
        'annual' or 'quarterly'.
    store_dir : Path, optional
    max_staleness_days : int
        If the best record is older than this many days before as_of, return None.
        Default 180 days (~2 reporting cycles). Set 0 to disable.

    Returns
    -------
    pd.Series or None
        Most recent on-or-before-as_of record, or None if none available /
        too stale.
    """
    df = load_fundamentals(source=source, store_dir=store_dir, tickers=[ticker])
    if df.empty:
        return None

    df = df[df["period_type"] == period_type].copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"]).dt.date

    # Point-in-time filter: only use data known at as_of
    eligible = df[df["filing_date"] <= as_of]
    if eligible.empty:
        return None

    best = eligible.sort_values("filing_date").iloc[-1]

    if max_staleness_days > 0:
        staleness = (as_of - best["filing_date"]).days
        if staleness > max_staleness_days:
            return None

    return best


def is_cache_populated(source: str, store_dir: Optional[Path] = None) -> bool:
    """Return True if a non-empty Parquet cache exists for this source."""
    if store_dir is None:
        from config import CACHE_DIR
        store_dir = CACHE_DIR / "fundamentals"
    path = _store_path(store_dir, source)
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path)
        return not df.empty
    except Exception:
        return False
