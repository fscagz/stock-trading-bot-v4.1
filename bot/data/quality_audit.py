"""
Data quality audit for the systematic pipeline.

Checks every (ticker, rebalance_date) cell in the feature matrix for:
  - Missing values
  - Stale fundamentals (filing_date older than max_staleness_days)
  - Implausible values (negative market cap, P/E < 0, etc.)
  - Fiscal year misalignment (filing covers wrong period)

Results are written to a CSV log so they can be reviewed before modeling.
All checks are non-blocking (they log, not raise), unless a hard threshold
is exceeded — in which case the audit returns a summary flag.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Anomaly record
# ---------------------------------------------------------------------------

@dataclass
class Anomaly:
    ticker: str
    rebalance_date: date
    check: str          # short identifier, e.g. 'missing_value'
    column: str         # feature / column name where the issue appears
    detail: str         # human-readable description
    severity: str       # 'warning' | 'error'


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_missing_values(
    ticker: str,
    rebalance_date: date,
    row: pd.Series,
    required_columns: Optional[List[str]] = None,
) -> List[Anomaly]:
    """Flag NaN values in a feature row."""
    anomalies = []
    cols = required_columns if required_columns is not None else row.index.tolist()
    for col in cols:
        if col not in row.index:
            anomalies.append(Anomaly(
                ticker=ticker, rebalance_date=rebalance_date,
                check="missing_column", column=col,
                detail=f"Column '{col}' absent from row",
                severity="error",
            ))
        elif pd.isna(row[col]):
            anomalies.append(Anomaly(
                ticker=ticker, rebalance_date=rebalance_date,
                check="missing_value", column=col,
                detail=f"NaN in '{col}'",
                severity="warning",
            ))
    return anomalies


def check_stale_fundamentals(
    ticker: str,
    rebalance_date: date,
    filing_date: Optional[date],
    max_staleness_days: int = 180,
) -> List[Anomaly]:
    """Flag fundamentals whose filing_date is too old."""
    if filing_date is None:
        return [Anomaly(
            ticker=ticker, rebalance_date=rebalance_date,
            check="stale_fundamental", column="filing_date",
            detail="filing_date is None (no fundamental data available)",
            severity="error",
        )]
    staleness = (rebalance_date - filing_date).days
    if staleness > max_staleness_days:
        return [Anomaly(
            ticker=ticker, rebalance_date=rebalance_date,
            check="stale_fundamental", column="filing_date",
            detail=f"filing_date={filing_date} is {staleness} days before rebalance "
                   f"(limit={max_staleness_days})",
            severity="warning" if staleness <= max_staleness_days * 1.5 else "error",
        )]
    return []


def check_implausible_values(
    ticker: str,
    rebalance_date: date,
    row: pd.Series,
) -> List[Anomaly]:
    """
    Flag financially implausible values.

    Rules:
      - pe_ratio outside (-200, 2000): suspicious (negative PE can be real but rare)
      - ev_ebitda outside (-50, 1000)
      - gross_margin outside (-1.0, 1.0): must be a ratio
      - roe outside (-10.0, 10.0): 1000% ROE is data error
      - debt_to_equity < 0: debt cannot be negative
      - revenue_growth / eps_growth outside (-5.0, 20.0): >2000% growth is suspicious
    """
    anomalies = []

    def _check(col: str, lo: float, hi: float, sev: str = "warning") -> None:
        if col not in row.index or pd.isna(row[col]):
            return
        v = float(row[col])
        if not (lo <= v <= hi):
            anomalies.append(Anomaly(
                ticker=ticker, rebalance_date=rebalance_date,
                check="implausible_value", column=col,
                detail=f"{col}={v:.4g} outside expected range [{lo}, {hi}]",
                severity=sev,
            ))

    _check("pe_ratio", -200.0, 2000.0)
    _check("ev_ebitda", -50.0, 1000.0)
    _check("gross_margin", -1.0, 1.0, sev="error")
    _check("roe", -10.0, 10.0)
    _check("debt_to_equity", 0.0, 1e6)  # negative D/E is a data error
    _check("revenue_growth", -5.0, 20.0)
    _check("eps_growth", -5.0, 20.0)
    _check("fcf_yield", -5.0, 5.0)

    return anomalies


def check_fiscal_year_alignment(
    ticker: str,
    rebalance_date: date,
    period_end_date: Optional[date],
    filing_date: Optional[date],
    max_report_to_filing_days: int = 180,
) -> List[Anomaly]:
    """
    Flag cases where the gap between period_end_date and filing_date is
    suspiciously large (indicates wrong period being used) or where the
    filing_date is before the period_end_date (physically impossible).
    """
    anomalies = []
    if period_end_date is None or filing_date is None:
        return anomalies
    gap = (filing_date - period_end_date).days
    if gap < 0:
        anomalies.append(Anomaly(
            ticker=ticker, rebalance_date=rebalance_date,
            check="fiscal_alignment", column="period_end_date",
            detail=f"filing_date={filing_date} is BEFORE period_end_date={period_end_date} "
                   f"(gap={gap} days) — data error",
            severity="error",
        ))
    elif gap > max_report_to_filing_days:
        anomalies.append(Anomaly(
            ticker=ticker, rebalance_date=rebalance_date,
            check="fiscal_alignment", column="period_end_date",
            detail=f"filing_date={filing_date} is {gap} days after period_end_date={period_end_date} "
                   f"(limit={max_report_to_filing_days}) — possibly wrong fiscal period",
            severity="warning",
        ))
    return anomalies


# ---------------------------------------------------------------------------
# Batch audit
# ---------------------------------------------------------------------------

def audit_feature_matrix(
    feature_matrix: pd.DataFrame,
    rebalance_date: date,
    filing_dates: Optional[Dict[str, date]] = None,
    period_end_dates: Optional[Dict[str, date]] = None,
    required_columns: Optional[List[str]] = None,
    max_staleness_days: int = 180,
) -> List[Anomaly]:
    """
    Run all quality checks on a feature matrix for one rebalance date.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Rows = tickers, columns = features.
    rebalance_date : date
    filing_dates : dict, optional
        {ticker: filing_date} for fundamental staleness check.
    period_end_dates : dict, optional
        {ticker: period_end_date} for fiscal alignment check.
    required_columns : list, optional
        Columns that must be non-null. Defaults to all columns.
    max_staleness_days : int

    Returns
    -------
    list of Anomaly
    """
    all_anomalies: List[Anomaly] = []
    filing_dates = filing_dates or {}
    period_end_dates = period_end_dates or {}

    for ticker, row in feature_matrix.iterrows():
        all_anomalies += check_missing_values(
            ticker, rebalance_date, row, required_columns
        )
        all_anomalies += check_implausible_values(ticker, rebalance_date, row)
        if filing_dates or period_end_dates:
            all_anomalies += check_stale_fundamentals(
                ticker, rebalance_date,
                filing_dates.get(ticker),
                max_staleness_days,
            )
            all_anomalies += check_fiscal_year_alignment(
                ticker, rebalance_date,
                period_end_dates.get(ticker),
                filing_dates.get(ticker),
            )

    return all_anomalies


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_anomaly_log(
    anomalies: List[Anomaly],
    log_path: Path,
    append: bool = True,
) -> None:
    """
    Write anomalies to a CSV log file.

    Parameters
    ----------
    anomalies : list of Anomaly
    log_path : Path
    append : bool
        If True, append to existing file. If False, overwrite.
    """
    if not anomalies:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and log_path.exists() else "w"
    with open(log_path, mode, newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ticker", "rebalance_date", "check", "column", "detail", "severity"]
        )
        if mode == "w":
            writer.writeheader()
        for a in anomalies:
            writer.writerow({
                "ticker": a.ticker,
                "rebalance_date": a.rebalance_date.isoformat(),
                "check": a.check,
                "column": a.column,
                "detail": a.detail,
                "severity": a.severity,
            })


def summarise_anomalies(anomalies: List[Anomaly]) -> Dict[str, int]:
    """
    Return counts by check type and severity.

    Returns
    -------
    dict, e.g. {'missing_value:warning': 12, 'implausible_value:warning': 3, ...}
    """
    counts: Dict[str, int] = {}
    for a in anomalies:
        key = f"{a.check}:{a.severity}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def audit_passes(
    anomalies: List[Anomaly],
    max_error_rate: float = 0.05,
    n_tickers: int = 1,
) -> Tuple[bool, str]:
    """
    Decide whether data quality is good enough to proceed.

    Fails if the fraction of tickers with at least one ERROR-level anomaly
    exceeds max_error_rate.

    Parameters
    ----------
    anomalies : list of Anomaly
    max_error_rate : float
        Maximum tolerable fraction of tickers with errors (default 5%).
    n_tickers : int
        Total number of tickers in the universe (denominator for error rate).

    Returns
    -------
    (passed: bool, message: str)
    """
    error_tickers = {a.ticker for a in anomalies if a.severity == "error"}
    rate = len(error_tickers) / n_tickers if n_tickers > 0 else 0.0
    if rate > max_error_rate:
        return False, (
            f"Quality audit FAILED: {len(error_tickers)}/{n_tickers} tickers "
            f"({rate:.1%}) have errors — threshold is {max_error_rate:.1%}."
        )
    return True, (
        f"Quality audit PASSED: {len(error_tickers)}/{n_tickers} tickers "
        f"({rate:.1%}) have errors."
    )
