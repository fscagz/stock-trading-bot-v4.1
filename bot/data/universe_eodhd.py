"""
EODHD historical S&P 500 constituent loader.

Provides a survivorship-bias-free universe for any historical rebalance date
by reconstructing index membership from EODHD's constituent change history.

Activation
----------
Set BOT_USE_EODHD=true and EODHD_API_KEY=<your_key> in your environment (or .env).
Obtain a key at https://eodhd.com — the "All-World" plan (~$20–50/mo) includes
historical constituent data.

How it works
------------
EODHD's /historical-constituent-changes endpoint returns every addition and
deletion from the S&P 500 with the effective date.  Starting from a known
baseline of current constituents, we replay the change history in reverse to
reconstruct membership at any historical date.

Usage
-----
    from data.universe_eodhd import load_sp500_constituents_as_of
    tickers = load_sp500_constituents_as_of(date(2015, 6, 30))

The module also provides bulk_build_snapshots() to pre-generate universe
snapshot CSVs for a list of rebalance dates (used during initial setup).
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

_BASE_URL = "https://eodhd.com/api"
_INDEX_CODE = "GSPC.INDX"  # S&P 500


# ---------------------------------------------------------------------------
# Internal fetch helpers
# ---------------------------------------------------------------------------

def _get(endpoint: str, api_key: str, params: Optional[Dict[str, str]] = None) -> dict | list:
    """GET a JSON endpoint from EODHD, returning parsed data."""
    base = f"{_BASE_URL}/{endpoint}?api_token={api_key}&fmt=json"
    if params:
        base += "&" + "&".join(f"{k}={v}" for k, v in params.items())
    req = urllib.request.Request(base, headers={"User-Agent": "bot/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _normalize_ticker(ticker: str) -> str:
    """Convert EODHD ticker format to yfinance format (e.g. BRK.B → BRK-B)."""
    return ticker.replace(".", "-")


# ---------------------------------------------------------------------------
# Constituent change history
# ---------------------------------------------------------------------------

def fetch_constituent_changes(api_key: str) -> pd.DataFrame:
    """
    Fetch all historical S&P 500 constituent additions and deletions from EODHD.

    Returns
    -------
    pd.DataFrame with columns: date, ticker, change_type ('add' | 'remove')
    Sorted ascending by date.
    """
    data = _get(f"historical-constituent-changes/{_INDEX_CODE}", api_key)
    if not data:
        return pd.DataFrame(columns=["date", "ticker", "change_type"])

    rows = []
    for record in data:
        # EODHD returns: {"date": "YYYY-MM-DD", "added": [...], "removed": [...]}
        record_date = date.fromisoformat(record["date"])
        for ticker in record.get("added", []) or []:
            rows.append({"date": record_date, "ticker": _normalize_ticker(ticker), "change_type": "add"})
        for ticker in record.get("removed", []) or []:
            rows.append({"date": record_date, "ticker": _normalize_ticker(ticker), "change_type": "remove"})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("date").reset_index(drop=True)


def fetch_current_constituents(api_key: str) -> List[str]:
    """
    Fetch today's S&P 500 constituents from EODHD as the reconstruction baseline.

    Returns
    -------
    List of ticker strings in yfinance format.
    """
    data = _get(f"fundamentals/{_INDEX_CODE}", api_key)
    components = data.get("General", {}).get("Components", {})
    tickers = [_normalize_ticker(v["Code"]) for v in components.values()]
    return sorted(tickers)


# ---------------------------------------------------------------------------
# Point-in-time reconstruction
# ---------------------------------------------------------------------------

def reconstruct_constituents_as_of(
    as_of: date,
    current_constituents: List[str],
    changes: pd.DataFrame,
) -> List[str]:
    """
    Reconstruct S&P 500 membership as of `as_of` by replaying change history.

    Algorithm: start with current members, then reverse all changes that
    happened *after* `as_of`:
      - An addition after as_of → remove it (it wasn't in yet)
      - A removal after as_of → add it back (it hadn't left yet)

    Parameters
    ----------
    as_of : date
        Target historical date.
    current_constituents : list of str
        Present-day S&P 500 tickers (yfinance format).
    changes : pd.DataFrame
        Output of fetch_constituent_changes().

    Returns
    -------
    Sorted list of ticker strings as of `as_of`.
    """
    members: Set[str] = set(current_constituents)

    # Replay changes strictly after as_of, in reverse chronological order
    future_changes = changes[changes["date"] > as_of].sort_values("date", ascending=False)

    for _, row in future_changes.iterrows():
        ticker = row["ticker"]
        if row["change_type"] == "add":
            # Was added after as_of → wasn't in the index yet
            members.discard(ticker)
        else:  # remove
            # Was removed after as_of → was still in the index
            members.add(ticker)

    return sorted(members)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_sp500_constituents_as_of(
    as_of: date,
    api_key: Optional[str] = None,
) -> List[str]:
    """
    Return the S&P 500 constituent list as of `as_of` using EODHD data.

    Parameters
    ----------
    as_of : date
        Historical rebalance date.
    api_key : str, optional
        EODHD API key. Falls back to EODHD_API_KEY environment variable.

    Returns
    -------
    Sorted list of ticker strings in yfinance format.

    Raises
    ------
    ValueError
        If the API key is missing.
    RuntimeError
        If EODHD returns empty change history (likely an API key / plan issue).
    """
    import os
    if api_key is None:
        api_key = os.getenv("EODHD_API_KEY", "")
    if not api_key:
        raise ValueError("EODHD_API_KEY is not set. Obtain a key at https://eodhd.com.")

    current = fetch_current_constituents(api_key)
    changes = fetch_constituent_changes(api_key)

    if changes.empty:
        raise RuntimeError(
            "EODHD returned no constituent change history. Check your API plan — "
            "historical constituent data requires the All-World plan or higher."
        )

    return reconstruct_constituents_as_of(as_of, current, changes)


def bulk_build_snapshots(
    rebalance_dates: List[date],
    snapshot_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    verbose: bool = True,
    sleep_seconds: float = 0.5,
) -> Dict[date, Path]:
    """
    Pre-generate and save universe snapshot CSVs for a list of rebalance dates.

    This is the recommended one-time setup step for historical backtesting.
    Run it once to populate data/universe_snapshots/ before running any backtest.

    Parameters
    ----------
    rebalance_dates : list of date
        Historical rebalance dates to generate snapshots for.
    snapshot_dir : Path, optional
        Destination directory. Defaults to config.UNIVERSE_SNAPSHOT_DIR.
    api_key : str, optional
        EODHD API key. Defaults to config.EODHD_API_KEY.
    verbose : bool
        Print progress.
    sleep_seconds : float
        Pause between API calls to respect rate limits.

    Returns
    -------
    dict mapping date → Path of written CSV.
    """
    import os
    if api_key is None:
        api_key = os.getenv("EODHD_API_KEY", "")
    if not api_key:
        raise ValueError("EODHD_API_KEY is not set.")
    if snapshot_dir is None:
        snapshot_dir = Path("data/universe_snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[eodhd] Fetching current constituents and change history...")
    current = fetch_current_constituents(api_key)
    changes = fetch_constituent_changes(api_key)
    if changes.empty:
        raise RuntimeError("EODHD returned no constituent change history.")

    written: Dict[date, Path] = {}
    for i, d in enumerate(sorted(rebalance_dates)):
        tickers = reconstruct_constituents_as_of(d, current, changes)
        path = snapshot_dir / f"sp500_{d.isoformat()}.csv"
        pd.DataFrame({"ticker": tickers}).to_csv(path, index=False)
        written[d] = path
        if verbose:
            print(f"[eodhd] {i + 1}/{len(rebalance_dates)}  {d}  →  {len(tickers)} tickers  →  {path.name}")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return written
