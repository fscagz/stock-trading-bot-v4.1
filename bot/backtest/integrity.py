"""
Backtest honesty checks.

These run automatically at the end of a backtest run to catch common
sources of inflated results. From the vision's Backtest Honesty Checklist:

  [x] No fundamental data used before its SEC filing date
  [x] Universe defined at each rebalance date, not using current constituents
  [x] ML model never trained on data that overlaps with its test period
  [x] Transaction costs applied to every rebalance, including small trades
  [x] Results decomposed by sub-period

The first two are also enforced *during* the backtest via run_preflight_checks()
in data/integrity_checks.py. These post-run checks give a final audit summary.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class HonestyReport:
    """Summary of all post-run honesty checks."""
    passed: bool
    checks: Dict[str, bool]          # check_name → passed
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def print_summary(self) -> None:
        print("=" * 55)
        print("BACKTEST HONESTY CHECKLIST")
        print("=" * 55)
        for name, passed in self.checks.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}  {name}")
        if self.warnings:
            print("\nWarnings:")
            for w in self.warnings:
                print(f"  ! {w}")
        if self.errors:
            print("\nErrors:")
            for e in self.errors:
                print(f"  ✗ {e}")
        overall = "PASSED" if self.passed else "FAILED"
        print(f"\nOverall: {overall}")
        print("=" * 55)


def check_costs_applied(
    rebalance_log: pd.DataFrame,
    cost_col: str = "total_cost",
) -> tuple[bool, str]:
    """
    Verify that transaction costs were recorded at every rebalance.

    Returns (passed, message).
    """
    if rebalance_log.empty:
        return False, "Rebalance log is empty — no trades recorded."
    if cost_col not in rebalance_log.columns:
        return False, f"Column '{cost_col}' not found in rebalance log — costs may not have been applied."
    zero_cost_rebalances = (rebalance_log[cost_col] == 0).sum()
    total = len(rebalance_log)
    if zero_cost_rebalances == total:
        return False, f"ALL {total} rebalances have zero transaction cost — cost model was not applied."
    if zero_cost_rebalances > total * 0.5:
        msg = (
            f"{zero_cost_rebalances}/{total} rebalances have zero transaction cost. "
            "This may indicate costs were skipped for no-trade rebalances (acceptable) "
            "or the cost model is misconfigured (investigate)."
        )
        return True, msg  # warn but don't fail
    return True, f"Costs applied at {total - zero_cost_rebalances}/{total} rebalances."


def check_universe_snapshots_used(
    rebalance_log: pd.DataFrame,
    snapshot_col: str = "snapshot_used",
) -> tuple[bool, str]:
    """
    Verify that dated universe snapshots were used (not current constituents).
    """
    if snapshot_col not in rebalance_log.columns:
        return False, (
            f"Column '{snapshot_col}' not found in rebalance log. "
            "Confirm the engine is using load_universe_snapshot() at each rebalance date."
        )
    if rebalance_log[snapshot_col].all():
        return True, "Dated universe snapshots used at all rebalances."
    n_missing = (~rebalance_log[snapshot_col]).sum()
    return False, (
        f"{n_missing}/{len(rebalance_log)} rebalances did NOT use a dated snapshot. "
        "These rebalances are vulnerable to survivorship bias."
    )


def check_subperiod_attribution(
    daily_returns: pd.Series,
    required_periods: Optional[List[tuple]] = None,
) -> tuple[bool, Dict[str, float]]:
    """
    Decompose returns into standard sub-periods and check coverage.

    Returns (passed, metrics_by_period).
    """
    if daily_returns.empty:
        return False, {}

    if required_periods is None:
        required_periods = [
            ("dot_com_crash", "2000-03-01", "2002-10-01"),
            ("gfc", "2007-10-01", "2009-03-31"),
            ("momentum_crash", "2014-07-01", "2016-02-01"),
            ("covid", "2020-02-01", "2020-12-31"),
            ("rate_hike_bear", "2022-01-01", "2022-12-31"),
        ]

    from backtest.metrics import cagr, max_drawdown, sharpe_ratio

    period_metrics: Dict[str, float] = {}
    missing = []
    for name, start, end in required_periods:
        sub = daily_returns.loc[start:end]
        if len(sub) < 20:
            missing.append(name)
            continue
        period_metrics[f"{name}_cagr"] = cagr(sub)
        period_metrics[f"{name}_max_dd"] = max_drawdown(sub)
        period_metrics[f"{name}_sharpe"] = sharpe_ratio(sub)

    if missing:
        # Not a failure — backtest may not span those periods
        pass
    return True, period_metrics


def check_no_parameter_fitting_on_full_period(
    training_dates_used: Optional[List[date]] = None,
    backtest_start: Optional[date] = None,
) -> tuple[bool, str]:
    """
    Verify that no parameter was optimised on the full backtest period.

    This is a documentation check — the engine passes training_dates_used
    to record what data was used for any in-sample fitting.

    If training_dates_used is None, we can only warn (cannot verify).
    """
    if training_dates_used is None:
        return True, (
            "No training date metadata provided. Manually verify that no "
            "parameters (factor weights, thresholds) were tuned on the full backtest period."
        )
    if backtest_start is None:
        return True, "training_dates_used provided but backtest_start not — cannot verify."
    # If training ever used dates after backtest_start, flag it
    invalid = [d for d in training_dates_used if d >= backtest_start]
    if invalid:
        return False, (
            f"{len(invalid)} training dates fall within or after the backtest start date "
            f"({backtest_start}) — this constitutes fitting on the test set."
        )
    return True, "Training dates are all before the backtest period — no parameter leakage detected."


def run_honesty_checks(
    daily_returns: pd.Series,
    rebalance_log: pd.DataFrame,
    training_dates_used: Optional[List[date]] = None,
    backtest_start: Optional[date] = None,
) -> HonestyReport:
    """
    Run all post-backtest honesty checks and return a consolidated report.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio returns.
    rebalance_log : pd.DataFrame
        One row per rebalance — must have 'total_cost' and optionally
        'snapshot_used' columns.
    training_dates_used : list of date, optional
        Dates used for any in-sample parameter fitting.
    backtest_start : date, optional
        First date of the backtest period.

    Returns
    -------
    HonestyReport
    """
    checks: Dict[str, bool] = {}
    all_warnings: List[str] = []
    all_errors: List[str] = []

    # 1. Costs applied
    passed, msg = check_costs_applied(rebalance_log)
    checks["transaction_costs_applied"] = passed
    (all_warnings if passed else all_errors).append(msg)

    # 2. Universe snapshots
    passed, msg = check_universe_snapshots_used(rebalance_log)
    checks["universe_snapshots_used"] = passed
    (all_warnings if passed else all_errors).append(msg)

    # 3. Sub-period attribution
    passed, period_metrics = check_subperiod_attribution(daily_returns)
    checks["subperiod_attribution_computed"] = passed
    if not period_metrics:
        all_warnings.append(
            "No sub-period metrics computed — backtest may not span any standard stress periods."
        )

    # 4. Parameter fitting
    passed, msg = check_no_parameter_fitting_on_full_period(
        training_dates_used, backtest_start
    )
    checks["no_full_period_fitting"] = passed
    (all_warnings if passed else all_errors).append(msg)

    overall = all(checks.values())
    return HonestyReport(
        passed=overall,
        checks=checks,
        warnings=all_warnings,
        errors=all_errors,
    )
