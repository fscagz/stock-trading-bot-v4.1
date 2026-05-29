"""
Data integrity checks — hard assertions before any backtest result is trusted.

Two categories of check:

1. Look-ahead bias test
   For every (ticker, rebalance_date) row in a feature matrix, assert that
   no fundamental data point has a filing_date AFTER the rebalance_date.
   A violation means future information leaked into the feature vector —
   backtest returns are fictitious.

2. Survivorship bias audit
   For a set of historical rebalance dates, compare the dated universe
   snapshots used in the backtest against the current S&P 500 constituent
   list. Tickers that appear in the current index but NOT in the historical
   snapshot are survivorship-biased additions — stocks that joined the index
   (or survived) after the historical date.

Both checks FAIL LOUDLY (raise IntegrityError) when violations are found.
They are designed to be called automatically by the backtest engine before
results are reported. Never suppress these errors without investigating.

Usage
-----
    from data.integrity_checks import assert_no_lookahead, survivorship_bias_report

    # Before reporting any backtest result:
    assert_no_lookahead(feature_matrix, rebalance_date, filing_dates)
    report = survivorship_bias_report(historical_snapshots, current_tickers)
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


class IntegrityError(Exception):
    """Raised when a data integrity violation is detected."""


# ---------------------------------------------------------------------------
# 1. Look-ahead bias checks
# ---------------------------------------------------------------------------

def check_lookahead_single(
    ticker: str,
    rebalance_date: date,
    filing_date: Optional[date],
) -> Optional[str]:
    """
    Check one (ticker, rebalance_date, filing_date) triple for look-ahead bias.

    Returns
    -------
    str (violation message) or None if clean.
    """
    if filing_date is None:
        return None  # No filing date → no fundamental data used → no look-ahead risk
    if filing_date > rebalance_date:
        return (
            f"LOOK-AHEAD BIAS: ticker={ticker}, rebalance_date={rebalance_date}, "
            f"filing_date={filing_date} — filing is {(filing_date - rebalance_date).days} "
            f"days AFTER rebalance date."
        )
    return None


def check_lookahead_batch(
    rebalance_date: date,
    filing_dates: Dict[str, Optional[date]],
) -> List[str]:
    """
    Check all (ticker, filing_date) pairs for one rebalance date.

    Parameters
    ----------
    rebalance_date : date
    filing_dates : dict {ticker: filing_date or None}

    Returns
    -------
    list of violation messages (empty if clean).
    """
    violations = []
    for ticker, fd in filing_dates.items():
        msg = check_lookahead_single(ticker, rebalance_date, fd)
        if msg:
            violations.append(msg)
    return violations


def assert_no_lookahead(
    rebalance_date: date,
    filing_dates: Dict[str, Optional[date]],
    context: str = "",
) -> None:
    """
    Hard assertion: raise IntegrityError if ANY filing_date is after rebalance_date.

    Call this inside the backtest loop before building the feature matrix.

    Parameters
    ----------
    rebalance_date : date
    filing_dates : dict {ticker: filing_date or None}
        The filing dates used when building the feature matrix at this rebalance date.
    context : str
        Optional description to include in the error message (e.g. backtest run name).

    Raises
    ------
    IntegrityError
        If any look-ahead violation is detected.
    """
    violations = check_lookahead_batch(rebalance_date, filing_dates)
    if violations:
        header = f"[integrity_checks] Look-ahead bias detected{' in ' + context if context else ''}:"
        detail = "\n  ".join(violations[:20])  # show up to 20 violations
        if len(violations) > 20:
            detail += f"\n  ... and {len(violations) - 20} more."
        raise IntegrityError(f"{header}\n  {detail}")


def scan_feature_matrix_for_lookahead(
    feature_matrix: pd.DataFrame,
    rebalance_date: date,
    filing_date_col: str = "filing_date",
) -> List[str]:
    """
    Scan a feature matrix that contains a filing_date column for look-ahead
    violations (rather than passing a separate dict).

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Rows = tickers; must contain filing_date_col.
    rebalance_date : date
    filing_date_col : str

    Returns
    -------
    list of violation messages.
    """
    if filing_date_col not in feature_matrix.columns:
        return []  # No filing date column → cannot check → caller must handle
    violations = []
    for ticker, row in feature_matrix.iterrows():
        fd = row[filing_date_col]
        if pd.isna(fd):
            continue
        fd = pd.Timestamp(fd).date() if not isinstance(fd, date) else fd
        msg = check_lookahead_single(str(ticker), rebalance_date, fd)
        if msg:
            violations.append(msg)
    return violations


# ---------------------------------------------------------------------------
# 2. Survivorship bias audit
# ---------------------------------------------------------------------------

def survivorship_bias_report(
    historical_snapshot: List[str],
    current_tickers: List[str],
    snapshot_date: Optional[date] = None,
) -> dict:
    """
    Compare a historical universe snapshot against the current constituent list.

    Tickers in current_tickers but NOT in historical_snapshot were added to
    the index AFTER snapshot_date (or survived while others were delisted).
    If a backtest used current_tickers retroactively for snapshot_date, it
    included these stocks before they were eligible — survivorship bias.

    Parameters
    ----------
    historical_snapshot : list of str
        Tickers in the universe at the historical rebalance date.
    current_tickers : list of str
        Current S&P 500 constituents (from Wikipedia or EODHD today).
    snapshot_date : date, optional
        Label for the historical date (included in the report for readability).

    Returns
    -------
    dict with keys:
        snapshot_date           : date or None
        n_historical            : int — tickers in historical snapshot
        n_current               : int — tickers in current list
        biased_additions        : list[str] — in current but not historical
        n_biased_additions      : int
        biased_pct              : float — biased_additions / n_current
        delisted_or_removed     : list[str] — in historical but not current
        n_delisted_or_removed   : int
        summary                 : str — human-readable summary
    """
    hist_set: Set[str] = set(historical_snapshot)
    curr_set: Set[str] = set(current_tickers)

    biased = sorted(curr_set - hist_set)       # added after snapshot_date
    delisted = sorted(hist_set - curr_set)      # removed/delisted since snapshot_date

    biased_pct = len(biased) / len(curr_set) if curr_set else 0.0
    date_str = snapshot_date.isoformat() if snapshot_date else "unknown date"

    if len(biased) == 0:
        summary = (
            f"[survivorship] {date_str}: No survivorship bias detected "
            f"(historical={len(hist_set)}, current={len(curr_set)})."
        )
    else:
        summary = (
            f"[survivorship] {date_str}: {len(biased)} tickers ({biased_pct:.1%}) are in the "
            f"current index but NOT in the historical snapshot — they joined after {date_str}. "
            f"Using the current index retroactively would introduce survivorship bias for these stocks."
        )

    return {
        "snapshot_date": snapshot_date,
        "n_historical": len(hist_set),
        "n_current": len(curr_set),
        "biased_additions": biased,
        "n_biased_additions": len(biased),
        "biased_pct": biased_pct,
        "delisted_or_removed": delisted,
        "n_delisted_or_removed": len(delisted),
        "summary": summary,
    }


def assert_universe_snapshot_used(
    backtest_tickers: List[str],
    snapshot_tickers: List[str],
    rebalance_date: date,
    max_extra_pct: float = 0.0,
) -> None:
    """
    Assert that the backtest used universe tickers consistent with the snapshot.

    Raises IntegrityError if backtest_tickers contains stocks not in the
    snapshot (i.e., the backtest is using forward-looking universe information).

    Parameters
    ----------
    backtest_tickers : list of str
        Tickers actually used in the backtest at this rebalance date.
    snapshot_tickers : list of str
        Tickers from the dated universe snapshot for this date.
    rebalance_date : date
    max_extra_pct : float
        Tolerate up to this fraction of extra tickers (for minor data
        discrepancies). Default 0 = zero tolerance.

    Raises
    ------
    IntegrityError
        If extra tickers exceed max_extra_pct of the snapshot size.
    """
    snap_set = set(snapshot_tickers)
    extra = [t for t in backtest_tickers if t not in snap_set]
    extra_pct = len(extra) / len(snap_set) if snap_set else 0.0

    if extra_pct > max_extra_pct:
        raise IntegrityError(
            f"[integrity_checks] Survivorship bias risk at {rebalance_date}: "
            f"{len(extra)} backtest tickers ({extra_pct:.1%}) are not in the dated "
            f"universe snapshot: {extra[:10]}{'...' if len(extra) > 10 else ''}. "
            f"Use load_universe_snapshot() to get the correct historical universe."
        )


# ---------------------------------------------------------------------------
# 3. ML train/test date discipline
# ---------------------------------------------------------------------------

def assert_no_train_test_overlap(
    train_end: date,
    test_start: date,
    context: str = "",
) -> None:
    """
    Assert that the ML training period ends before the test period starts.

    An off-by-one error here is the most common source of catastrophic
    look-ahead bias in ML backtests.

    Parameters
    ----------
    train_end : date
        Last date included in the training set.
    test_start : date
        First date of the test (out-of-sample) period.
    context : str
        Optional label for the fold.

    Raises
    ------
    IntegrityError
        If train_end >= test_start.
    """
    if train_end >= test_start:
        raise IntegrityError(
            f"[integrity_checks] Train/test overlap{' in ' + context if context else ''}: "
            f"train_end={train_end} is not before test_start={test_start}. "
            f"Labels for the test period must not appear in the training set."
        )


# ---------------------------------------------------------------------------
# 4. Composite pre-backtest gate
# ---------------------------------------------------------------------------

def run_preflight_checks(
    rebalance_date: date,
    filing_dates: Dict[str, Optional[date]],
    backtest_tickers: List[str],
    snapshot_tickers: List[str],
    context: str = "",
) -> None:
    """
    Run all integrity checks before a single backtest rebalance step.

    Call this inside the backtest engine loop. Raises IntegrityError on the
    first violation found.

    Parameters
    ----------
    rebalance_date : date
    filing_dates : dict {ticker: filing_date or None}
    backtest_tickers : list of str — tickers used at this rebalance step
    snapshot_tickers : list of str — from the dated universe snapshot
    context : str — optional run label for error messages
    """
    assert_no_lookahead(rebalance_date, filing_dates, context)
    assert_universe_snapshot_used(backtest_tickers, snapshot_tickers, rebalance_date)
