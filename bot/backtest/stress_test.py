"""
Stress test runner.

Runs the same backtest pipeline on specific historical sub-periods known
for challenging conditions. From the vision:

  - 2000-03-01 to 2002-10-01  (dot-com crash)
  - 2007-10-01 to 2009-03-31  (Global Financial Crisis)
  - 2014-07-01 to 2016-02-01  (momentum crash — tests momentum factors specifically)
  - 2020-02-01 to 2020-12-31  (COVID crash and recovery)
  - 2022-01-01 to 2022-12-31  (rate-hike bear market)

A strategy that only performs well in bull markets is not robust. If alpha
exists only in one regime, it's likely regime-dependent luck, not true alpha.

Usage
-----
    from backtest.stress_test import run_stress_tests, print_stress_report

    results = run_stress_tests(engine, signal_fn, price_panel, fallback_universe)
    print_stress_report(results)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pandas as pd

from backtest.metrics import compute_metrics, cagr, max_drawdown, sharpe_ratio


# ---------------------------------------------------------------------------
# Predefined stress periods
# ---------------------------------------------------------------------------

STRESS_PERIODS = [
    ("dot_com_crash",     "2000-03-01", "2002-10-01"),
    ("gfc",               "2007-10-01", "2009-03-31"),
    ("momentum_crash",    "2014-07-01", "2016-02-01"),
    ("covid",             "2020-02-01", "2020-12-31"),
    ("rate_hike_bear",    "2022-01-01", "2022-12-31"),
]

# Non-stress (bull) periods for comparison
BULL_PERIODS = [
    ("post_gfc_bull",     "2009-04-01", "2011-12-31"),
    ("mid_cycle_bull",    "2016-03-01", "2019-12-31"),
    ("post_covid_bull",   "2021-01-01", "2021-12-31"),
]

ALL_PERIODS = STRESS_PERIODS + BULL_PERIODS


# ---------------------------------------------------------------------------
# Sub-period metrics extraction
# ---------------------------------------------------------------------------

@dataclass
class PeriodMetrics:
    name: str
    start: str
    end: str
    cagr: float
    max_drawdown: float
    sharpe: float
    n_days: int
    covered: bool   # False if backtest doesn't span this period


def extract_period_metrics(
    daily_returns: pd.Series,
    name: str,
    start: str,
    end: str,
) -> PeriodMetrics:
    """Extract metrics for one sub-period from a returns series."""
    sub = daily_returns.loc[start:end]
    if len(sub) < 20:
        return PeriodMetrics(
            name=name, start=start, end=end,
            cagr=float("nan"), max_drawdown=float("nan"), sharpe=float("nan"),
            n_days=len(sub), covered=False,
        )
    return PeriodMetrics(
        name=name, start=start, end=end,
        cagr=cagr(sub),
        max_drawdown=max_drawdown(sub),
        sharpe=sharpe_ratio(sub),
        n_days=len(sub),
        covered=True,
    )


# ---------------------------------------------------------------------------
# Main stress test runner
# ---------------------------------------------------------------------------

def run_stress_tests(
    engine,
    signal_fn: Callable,
    price_panel: Dict[str, pd.DataFrame],
    fallback_universe: Optional[List[str]] = None,
    periods: Optional[List] = None,
    verbose: bool = True,
) -> Dict[str, "BacktestResult"]:  # noqa: F821
    """
    Run the backtest engine over each stress period and return results.

    Parameters
    ----------
    engine : BacktestEngine
        Configured engine instance (cost model, snapshot_dir, etc.).
    signal_fn : callable
        Same interface as BacktestEngine.run().
    price_panel : dict {ticker: daily_df}
    fallback_universe : list, optional
    periods : list of (name, start, end), optional
        Override default STRESS_PERIODS.
    verbose : bool

    Returns
    -------
    dict {period_name: BacktestResult}
        Failed periods are omitted with a warning.
    """
    periods = periods or STRESS_PERIODS
    results = {}

    for name, start, end in periods:
        if verbose:
            print(f"\n[stress_test] Running period: {name} ({start} → {end})")
        try:
            result = engine.run(
                signal_fn=signal_fn,
                price_panel=price_panel,
                start_date=start,
                end_date=end,
                fallback_universe=fallback_universe,
                verbose=False,
            )
            results[name] = result
            if verbose:
                m = result.metrics
                print(
                    f"  CAGR={m.get('cagr', float('nan')):.1%}  "
                    f"MaxDD={m.get('max_drawdown', float('nan')):.1%}  "
                    f"Sharpe={m.get('sharpe', float('nan')):.2f}"
                )
        except Exception as e:
            warnings.warn(f"[stress_test] Period '{name}' failed: {e}")

    return results


def run_subperiod_attribution(
    daily_returns: pd.Series,
    periods: Optional[List] = None,
) -> Dict[str, PeriodMetrics]:
    """
    Extract sub-period metrics from an existing returns series (faster than
    re-running the engine per period — just slices a completed backtest).

    Parameters
    ----------
    daily_returns : pd.Series
        Full daily returns series from a completed backtest.
    periods : list of (name, start, end), optional

    Returns
    -------
    dict {name: PeriodMetrics}
    """
    periods = periods or ALL_PERIODS
    return {
        name: extract_period_metrics(daily_returns, name, start, end)
        for name, start, end in periods
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_stress_report(
    results: Dict[str, "BacktestResult"],  # noqa: F821
    benchmark_results: Optional[Dict[str, "BacktestResult"]] = None,  # noqa: F821
) -> None:
    """Print a formatted stress test summary table."""
    print("\n" + "=" * 75)
    print("STRESS TEST RESULTS")
    print("=" * 75)
    header = f"{'Period':<22} {'CAGR':>8} {'Max DD':>8} {'Sharpe':>8}"
    if benchmark_results:
        header += f"  {'Bench CAGR':>10} {'Bench DD':>9}"
    print(header)
    print("-" * 75)

    for name, start, end in STRESS_PERIODS + BULL_PERIODS:
        if name not in results:
            print(f"  {name:<20}  {'(not run)'}")
            continue
        r = results[name]
        m = r.metrics
        line = (
            f"  {name:<20}  "
            f"{m.get('cagr', float('nan')):>7.1%}  "
            f"{m.get('max_drawdown', float('nan')):>7.1%}  "
            f"{m.get('sharpe', float('nan')):>7.2f}"
        )
        if benchmark_results and name in benchmark_results:
            bm = benchmark_results[name].metrics
            line += (
                f"  {bm.get('cagr', float('nan')):>9.1%}  "
                f"{bm.get('max_drawdown', float('nan')):>8.1%}"
            )
        print(line)

    print("=" * 75)
    print(
        "\nNote: Alpha that exists only in bull periods is regime-dependent. "
        "Check drawdowns in stress periods against your 25% max drawdown limit."
    )


def flag_regime_dependent_alpha(
    results: Dict[str, "BacktestResult"],  # noqa: F821
) -> List[str]:
    """
    Detect if strategy alpha is concentrated in one regime.

    Returns a list of warning strings if:
      - All stress periods have negative CAGR
      - All bull periods have strongly positive CAGR
      - The gap is large (possible regime-dependent alpha)
    """
    warnings_list = []

    stress_cagrs = [
        results[n].metrics.get("cagr", float("nan"))
        for n, _, _ in STRESS_PERIODS
        if n in results
    ]
    bull_cagrs = [
        results[n].metrics.get("cagr", float("nan"))
        for n, _, _ in BULL_PERIODS
        if n in results
    ]

    stress_cagrs = [c for c in stress_cagrs if not pd.isna(c)]
    bull_cagrs = [c for c in bull_cagrs if not pd.isna(c)]

    if stress_cagrs and all(c < 0 for c in stress_cagrs):
        warnings_list.append(
            "All stress periods show negative CAGR. Strategy may not provide "
            "protection during market downturns."
        )

    if stress_cagrs and bull_cagrs:
        avg_stress = sum(stress_cagrs) / len(stress_cagrs)
        avg_bull = sum(bull_cagrs) / len(bull_cagrs)
        if avg_bull - avg_stress > 0.20:
            warnings_list.append(
                f"Large performance gap: avg bull CAGR={avg_bull:.1%}, "
                f"avg stress CAGR={avg_stress:.1%} — alpha may be regime-dependent."
            )

    return warnings_list
