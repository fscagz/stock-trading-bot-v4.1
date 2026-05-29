"""
Investment universe for the systematic pipeline.

US large/mid cap (e.g. S&P 500), minimum liquidity (ADV) filter,
optional market cap filter.

Point-in-time safety
--------------------
get_tickers_sp500() fetches the *current* S&P 500 from Wikipedia.
It is ONLY safe for building the live/paper universe today.
DO NOT use it to construct historical backtest universes — doing so
introduces survivorship bias because you are selecting stocks that
survived to today's index.

For historical backtesting, use one of:
  - save_universe_snapshot() / load_universe_snapshot() to build and
    reuse dated snapshots accumulated over time.
  - universe_eodhd.load_sp500_constituents_as_of() when USE_EODHD=true,
    which uses EODHD's historical constituent change data.
"""

from __future__ import annotations

import io
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

from data.daily_loader import get_daily_batch

# Wikipedia blocks requests without a browser-like User-Agent (403 Forbidden).
WIKI_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Live-only helpers (current constituents — NOT for historical backtesting)
# ---------------------------------------------------------------------------

def get_tickers_sp500() -> List[str]:
    """
    Fetch the *current* S&P 500 ticker symbols from Wikipedia.

    WARNING: Returns today's constituents only. Using this for historical
    rebalance dates introduces survivorship bias. For backtesting use
    load_universe_snapshot() or universe_eodhd.

    Returns
    -------
    List of ticker strings (e.g. 'BRK-B' not 'BRK.B').
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={"User-Agent": WIKI_USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as response:
        raw = response.read()
    table = pd.read_html(io.BytesIO(raw))[0]
    tickers = table["Symbol"].tolist()
    return [t.replace(".", "-") for t in tickers]


def get_market_cap(symbol: str) -> Optional[float]:
    """
    Current market cap for a symbol (from yfinance). None if unavailable.

    WARNING: Returns today's market cap. Not suitable for historical filters.
    """
    try:
        t = yf.Ticker(symbol)
        info = t.info
        mc = info.get("marketCap")
        return float(mc) if mc is not None else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ADV helper (safe for any date range — uses price data)
# ---------------------------------------------------------------------------

def compute_adv_from_panel(
    symbol_to_df: dict[str, pd.DataFrame],
    min_trading_days: int = 10,
) -> dict[str, float]:
    """
    Average daily volume (ADV) per symbol from daily DataFrames.

    Parameters
    ----------
    symbol_to_df : dict[str, pd.DataFrame]
        Symbol -> daily OHLCV DataFrame (must have 'volume' column).
    min_trading_days : int
        Require at least this many days with volume to return an ADV.

    Returns
    -------
    dict[str, float]
        Symbol -> ADV (shares). Symbols with insufficient data are omitted.
    """
    out = {}
    for sym, df in symbol_to_df.items():
        if df.empty or "volume" not in df.columns:
            continue
        vol = df["volume"].dropna()
        if len(vol) < min_trading_days:
            continue
        out[sym] = float(vol.mean())
    return out


# ---------------------------------------------------------------------------
# Universe snapshot persistence
# ---------------------------------------------------------------------------

def _snapshot_path(snapshot_dir: Path, as_of: date) -> Path:
    return snapshot_dir / f"{as_of.isoformat()}.csv"


def save_universe_snapshot(
    tickers: List[str],
    as_of: date,
    snapshot_dir: Optional[Path] = None,
) -> Path:
    """
    Persist a dated universe snapshot to disk.

    Parameters
    ----------
    tickers : list of str
        Universe tickers valid on `as_of`.
    as_of : date
        The rebalance date this universe represents.
    snapshot_dir : Path, optional
        Directory to write CSV files. Defaults to config.UNIVERSE_SNAPSHOT_DIR.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    if snapshot_dir is None:
        from config import UNIVERSE_SNAPSHOT_DIR
        snapshot_dir = UNIVERSE_SNAPSHOT_DIR
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = _snapshot_path(snapshot_dir, as_of)
    pd.DataFrame({"ticker": sorted(tickers)}).to_csv(path, index=False)
    return path


def load_universe_snapshot(
    as_of: date,
    snapshot_dir: Optional[Path] = None,
    max_staleness_days: int = 45,
) -> Optional[List[str]]:
    """
    Load the most recent universe snapshot at or before `as_of`.

    Parameters
    ----------
    as_of : date
        The rebalance date. Returns the latest snapshot on or before this date.
    snapshot_dir : Path, optional
        Directory containing snapshot CSVs. Defaults to config.UNIVERSE_SNAPSHOT_DIR.
    max_staleness_days : int
        Raise if the best available snapshot is older than this many days.
        Set to 0 to disable the check.

    Returns
    -------
    list of str or None
        Sorted ticker list, or None if no snapshot exists.

    Raises
    ------
    ValueError
        If the best snapshot is older than max_staleness_days.
    """
    if snapshot_dir is None:
        from config import UNIVERSE_SNAPSHOT_DIR
        snapshot_dir = UNIVERSE_SNAPSHOT_DIR
    if not snapshot_dir.exists():
        return None

    available = list_universe_snapshots(snapshot_dir)
    if not available:
        return None

    # Most recent snapshot on or before as_of
    candidates = [d for d in available if d <= as_of]
    if not candidates:
        return None

    best = max(candidates)
    if max_staleness_days > 0 and (as_of - best).days > max_staleness_days:
        raise ValueError(
            f"Universe snapshot for {as_of} is stale: best available is {best} "
            f"({(as_of - best).days} days old, limit {max_staleness_days})."
        )

    path = _snapshot_path(snapshot_dir, best)
    df = pd.read_csv(path)
    return sorted(df["ticker"].tolist())


def list_universe_snapshots(snapshot_dir: Optional[Path] = None) -> List[date]:
    """
    Return all snapshot dates available on disk, sorted ascending.
    """
    if snapshot_dir is None:
        from config import UNIVERSE_SNAPSHOT_DIR
        snapshot_dir = UNIVERSE_SNAPSHOT_DIR
    if not snapshot_dir.exists():
        return []
    dates = []
    for p in snapshot_dir.glob("????-??-??.csv"):
        try:
            dates.append(date.fromisoformat(p.stem))
        except ValueError:
            pass
    return sorted(dates)


# ---------------------------------------------------------------------------
# Universe construction (live + snapshot saving)
# ---------------------------------------------------------------------------

def get_universe(
    source: str = "sp500",
    min_adv: Optional[int] = None,
    min_market_cap: Optional[float] = None,
    adv_lookback_days: int = 60,
    batch_size: int = 100,
    verbose: bool = False,
    save_snapshot: bool = False,
    snapshot_as_of: Optional[date] = None,
    snapshot_dir: Optional[Path] = None,
) -> List[str]:
    """
    Build the investable universe: tickers passing liquidity (and optional
    market cap) filter.

    WARNING: Uses the *current* S&P 500 constituent list as the candidate
    pool. Safe for live/paper use. For historical backtesting, rely on
    snapshots (save_snapshot=True accumulates them over time) or EODHD.

    Parameters
    ----------
    source : str
        Currently only "sp500".
    min_adv : int, optional
        Minimum average daily volume (shares).
    min_market_cap : float, optional
        Minimum market cap (requires one info call per symbol if set).
    adv_lookback_days : int
        Calendar days of price data used to compute ADV.
    batch_size : int
        Symbols per batch when fetching daily data for ADV.
    verbose : bool
        Print progress.
    save_snapshot : bool
        If True, persist the resulting universe as a dated snapshot.
    snapshot_as_of : date, optional
        Date label for the snapshot. Defaults to today.
    snapshot_dir : Path, optional
        Where to write snapshots. Defaults to config.UNIVERSE_SNAPSHOT_DIR.

    Returns
    -------
    List[str]
        Sorted list of ticker symbols passing all filters.
    """
    if source != "sp500":
        raise ValueError(f"Universe source '{source}' not implemented; use 'sp500'.")
    tickers = get_tickers_sp500()
    if not tickers:
        return []

    # Period for ADV: ~3 months for 60 days of trading
    period = "3mo" if adv_lookback_days <= 66 else "6mo"

    # Compute ADV in batches
    symbol_to_adv: dict[str, float] = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        if verbose:
            print(f"[universe] ADV batch {i // batch_size + 1}/{(len(tickers) - 1) // batch_size + 1}")
        panel = get_daily_batch(batch, period=period)
        advs = compute_adv_from_panel(panel, min_trading_days=10)
        symbol_to_adv.update(advs)

    # Filter by min_adv
    if min_adv is not None:
        tickers = [s for s in tickers if symbol_to_adv.get(s, 0) >= min_adv]
    else:
        tickers = [s for s in tickers if s in symbol_to_adv]

    # Optional market cap filter (slower: one request per symbol)
    if min_market_cap is not None and min_market_cap > 0:
        kept = []
        for s in tickers:
            mc = get_market_cap(s)
            if mc is not None and mc >= min_market_cap:
                kept.append(s)
        tickers = kept

    tickers = sorted(tickers)

    if save_snapshot:
        as_of = snapshot_as_of or date.today()
        path = save_universe_snapshot(tickers, as_of=as_of, snapshot_dir=snapshot_dir)
        if verbose:
            print(f"[universe] Snapshot saved → {path}")

    return tickers


def get_universe_from_config(
    verbose: bool = False,
    save_snapshot: bool = False,
    snapshot_as_of: Optional[date] = None,
) -> List[str]:
    """
    Build universe using config defaults (MIN_AVG_DAILY_VOLUME, MIN_MARKET_CAP,
    UNIVERSE_SOURCE).
    """
    from config import (
        UNIVERSE_SOURCE,
        MIN_AVG_DAILY_VOLUME,
        MIN_MARKET_CAP,
    )
    return get_universe(
        source=UNIVERSE_SOURCE,
        min_adv=MIN_AVG_DAILY_VOLUME,
        min_market_cap=MIN_MARKET_CAP,
        verbose=verbose,
        save_snapshot=save_snapshot,
        snapshot_as_of=snapshot_as_of,
    )


def should_refresh_universe(
    last_refresh_date: Optional[date],
    as_of: Optional[date] = None,
    quarterly: bool = True,
) -> bool:
    """
    Whether the universe should be refreshed (e.g. new quarter).

    Parameters
    ----------
    last_refresh_date : date or None
        Date of last universe refresh.
    as_of : date or None
        Current date; defaults to today.
    quarterly : bool
        If True, refresh when we've entered a new quarter since last_refresh_date.

    Returns
    -------
    bool
        True if we should refresh (never refreshed, or new quarter).
    """
    today = as_of or date.today()
    if last_refresh_date is None:
        return True
    if not quarterly:
        return False
    # New quarter: compare (year, quarter)
    y1, q1 = last_refresh_date.year, (last_refresh_date.month - 1) // 3 + 1
    y2, q2 = today.year, (today.month - 1) // 3 + 1
    return (y2, q2) > (y1, q1)
