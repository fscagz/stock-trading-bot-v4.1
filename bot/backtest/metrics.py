"""
Performance metrics for the backtesting framework.

Standard return-based metrics (CAGR, Sharpe, Sortino, drawdown, Calmar,
turnover, hit rate) plus information coefficient (IC) metrics specific to
cross-sectional factor models.

IC and IC t-statistic are first-class metrics here because they measure
whether the signal actually predicts future returns — the true test of
a factor model, independent of how well it happened to fit the backtest period.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Return-based metrics
# ---------------------------------------------------------------------------

def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compound Annual Growth Rate from a daily return series.

    Parameters
    ----------
    returns : pd.Series
        Daily returns (e.g. 0.01 = 1%).
    periods_per_year : int
        Trading days per year (252 for daily data).
    """
    n = len(returns)
    if n == 0:
        return float("nan")
    total_return = (1 + returns).prod()
    years = n / periods_per_year
    return float(total_return ** (1 / years) - 1)


def annualised_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised standard deviation of daily returns."""
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_annual: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Sharpe ratio: (mean return - risk_free) / std, annualised.

    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    risk_free_annual : float
        Annual risk-free rate (e.g. 0.05 = 5%).
    """
    rf_daily = (1 + risk_free_annual) ** (1 / periods_per_year) - 1
    excess = returns - rf_daily
    vol = excess.std()
    if vol == 0:
        return float("nan")
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_annual: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Sortino ratio: uses downside deviation instead of total std.
    """
    rf_daily = (1 + risk_free_annual) ** (1 / periods_per_year) - 1
    excess = returns - rf_daily
    downside = excess[excess < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else float("nan")
    if downside_std == 0 or np.isnan(downside_std):
        return float("nan")
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown as a negative decimal (e.g. -0.25 = -25%).
    """
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    return float(dd.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """CAGR / |max drawdown|."""
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return float("nan")
    return float(cagr(returns, periods_per_year) / mdd)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Return the full drawdown series (each point is DD from prior peak)."""
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    return (cum - rolling_max) / rolling_max


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with positive returns."""
    if len(returns) == 0:
        return float("nan")
    return float((returns > 0).mean())


def annualised_turnover(
    weight_snapshots: pd.DataFrame,
    periods_per_year: int = 12,
) -> float:
    """
    One-way annualised turnover from monthly weight snapshots.

    Parameters
    ----------
    weight_snapshots : pd.DataFrame
        Rows = rebalance dates, columns = tickers, values = weights.
    periods_per_year : int
        Number of rebalance periods per year (12 for monthly).

    Returns
    -------
    float
        One-way annual turnover (e.g. 0.6 = 60%).
    """
    if len(weight_snapshots) < 2:
        return float("nan")
    filled = weight_snapshots.fillna(0)
    deltas = filled.diff().abs().sum(axis=1)
    one_way_per_period = float(deltas.iloc[1:].mean()) / 2  # both sides counted in diff
    return one_way_per_period * periods_per_year


# ---------------------------------------------------------------------------
# Information Coefficient (IC) metrics
# ---------------------------------------------------------------------------

def compute_ic(
    signal: pd.Series,
    forward_return: pd.Series,
    method: str = "spearman",
) -> float:
    """
    Information Coefficient: rank correlation between signal and forward return
    at one rebalance date.

    Parameters
    ----------
    signal : pd.Series
        Signal values indexed by ticker. Higher = more bullish.
    forward_return : pd.Series
        Forward returns indexed by ticker (same horizon as the prediction).
    method : str
        'spearman' (default, rank-based, robust) or 'pearson'.

    Returns
    -------
    float
        IC in [-1, 1]. NaN if insufficient data.
    """
    common = signal.index.intersection(forward_return.index)
    if len(common) < 5:
        return float("nan")
    s = signal.loc[common].dropna()
    r = forward_return.loc[common].dropna()
    common2 = s.index.intersection(r.index)
    if len(common2) < 5:
        return float("nan")
    s, r = s.loc[common2], r.loc[common2]
    if method == "spearman":
        corr, _ = stats.spearmanr(s, r)
    else:
        corr, _ = stats.pearsonr(s, r)
    return float(corr)


def compute_ic_series(
    signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """
    Compute IC at every rebalance date.

    Parameters
    ----------
    signals : pd.DataFrame
        Rows = rebalance dates, columns = tickers, values = signal scores.
    forward_returns : pd.DataFrame
        Same shape as signals. forward_returns.loc[t] is the return from t
        to t+horizon for each ticker.
    method : str

    Returns
    -------
    pd.Series
        Index = rebalance dates, values = IC.
    """
    dates = signals.index.intersection(forward_returns.index)
    ic_vals = {}
    for d in dates:
        ic_vals[d] = compute_ic(signals.loc[d], forward_returns.loc[d], method)
    return pd.Series(ic_vals, name="ic")


def ic_tstat(ic_series: pd.Series) -> float:
    """
    t-statistic of the IC series: mean(IC) / (std(IC) / sqrt(N)).

    A t-stat > 2 indicates the mean IC is statistically significant
    (not noise). This is a required metric per the vision.

    Parameters
    ----------
    ic_series : pd.Series
        Monthly IC values.
    """
    clean = ic_series.dropna()
    n = len(clean)
    if n < 3:
        return float("nan")
    mean_ic = clean.mean()
    std_ic = clean.std()
    if std_ic == 0:
        return float("nan")
    return float(mean_ic / (std_ic / np.sqrt(n)))


def ic_decay(
    signals: pd.DataFrame,
    price_data: Dict[str, pd.DataFrame],
    rebalance_dates: pd.DatetimeIndex,
    horizons_days: tuple = (5, 10, 21, 42, 63),
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Compute IC at multiple forward horizons to produce an alpha decay curve.

    Parameters
    ----------
    signals : pd.DataFrame
        Rows = rebalance dates, columns = tickers.
    price_data : dict {ticker: daily OHLCV DataFrame}
        Full price panel (should cover the full backtest period).
    rebalance_dates : pd.DatetimeIndex
    horizons_days : tuple of int
        Forward windows in trading days (e.g. 5, 21, 63 for 1w/1m/3m).
    method : str

    Returns
    -------
    pd.DataFrame
        Columns = horizons (e.g. 'ic_5d', 'ic_21d'), index = rebalance dates.
        Each cell is the IC of the signal at that date for that horizon.
    """
    results = {}
    for h in horizons_days:
        ic_at_horizon = {}
        for d in rebalance_dates:
            if d not in signals.index:
                continue
            sig = signals.loc[d].dropna()
            tickers = sig.index.tolist()
            # Compute forward returns from d+1 to d+h
            fwd = {}
            for t in tickers:
                if t not in price_data or price_data[t].empty:
                    continue
                df = price_data[t]
                df_after = df[df.index > d]
                if len(df_after) < h:
                    continue
                price_start = df_after["close"].iloc[0]
                price_end = df_after["close"].iloc[min(h - 1, len(df_after) - 1)]
                if price_start > 0:
                    fwd[t] = price_end / price_start - 1
            if len(fwd) < 5:
                continue
            fwd_series = pd.Series(fwd)
            ic_at_horizon[d] = compute_ic(sig, fwd_series, method)
        results[f"ic_{h}d"] = ic_at_horizon

    df_out = pd.DataFrame(results)
    if not df_out.empty:
        df_out.index.name = "rebalance_date"
    return df_out


# ---------------------------------------------------------------------------
# Composite metrics report
# ---------------------------------------------------------------------------

def compute_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    ic_series: Optional[pd.Series] = None,
    weight_snapshots: Optional[pd.DataFrame] = None,
    risk_free_annual: float = 0.0,
    periods_per_year: int = 252,
    rebalances_per_year: int = 12,
) -> Dict[str, float]:
    """
    Compute the full metrics suite for a backtest result.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    benchmark_returns : pd.Series, optional
        Daily benchmark (e.g. SPY) returns for relative metrics.
    ic_series : pd.Series, optional
        Monthly IC series from compute_ic_series().
    weight_snapshots : pd.DataFrame, optional
        Monthly weight snapshots for turnover calculation.
    risk_free_annual : float
    periods_per_year : int
    rebalances_per_year : int

    Returns
    -------
    dict with keys matching the vision's target threshold table.
    """
    m: Dict[str, float] = {}

    m["cagr"] = cagr(returns, periods_per_year)
    m["annual_vol"] = annualised_volatility(returns, periods_per_year)
    m["sharpe"] = sharpe_ratio(returns, risk_free_annual, periods_per_year)
    m["sortino"] = sortino_ratio(returns, risk_free_annual, periods_per_year)
    m["max_drawdown"] = max_drawdown(returns)
    m["calmar"] = calmar_ratio(returns, periods_per_year)
    m["hit_rate"] = hit_rate(returns)

    if benchmark_returns is not None:
        common = returns.index.intersection(benchmark_returns.index)
        if len(common) > 10:
            excess = returns.loc[common] - benchmark_returns.loc[common]
            m["alpha_cagr"] = cagr(excess + 1, periods_per_year) - 1  # rough alpha
            m["benchmark_cagr"] = cagr(benchmark_returns.loc[common], periods_per_year)
            m["information_ratio"] = sharpe_ratio(excess, 0.0, periods_per_year)

    if weight_snapshots is not None and not weight_snapshots.empty:
        m["annual_turnover"] = annualised_turnover(weight_snapshots, rebalances_per_year)

    if ic_series is not None and not ic_series.empty:
        clean_ic = ic_series.dropna()
        m["ic_mean"] = float(clean_ic.mean())
        m["ic_std"] = float(clean_ic.std())
        m["ic_tstat"] = ic_tstat(ic_series)

    return m


def format_metrics(metrics: Dict[str, float]) -> str:
    """Return a formatted string summary of the metrics dict."""
    targets = {
        "cagr": ("> SPY", None),
        "sharpe": ("> 0.7", 0.7),
        "sortino": ("> 1.0", 1.0),
        "max_drawdown": ("> -25%", -0.25),
        "calmar": ("> 0.5", 0.5),
        "annual_turnover": ("< 150%", 1.5),
        "hit_rate": ("> 52%", 0.52),
        "ic_mean": ("> 0.05", 0.05),
        "ic_tstat": ("> 2.0", 2.0),
    }
    lines = ["=" * 50, "BACKTEST METRICS", "=" * 50]
    for key, val in sorted(metrics.items()):
        if np.isnan(val):
            lines.append(f"  {key:<25} {'N/A':>10}")
            continue
        # Format as percent for return/drawdown metrics, otherwise decimal
        pct_keys = {"cagr", "alpha_cagr", "benchmark_cagr", "annual_vol",
                    "max_drawdown", "hit_rate", "annual_turnover"}
        formatted = f"{val:.1%}" if key in pct_keys else f"{val:.3f}"
        target_str, threshold = targets.get(key, ("", None))
        if threshold is not None:
            if key == "max_drawdown":
                pass_fail = "✓" if val > threshold else "✗"
            else:
                pass_fail = "✓" if val >= threshold else "✗"
            lines.append(f"  {key:<25} {formatted:>10}  {pass_fail} {target_str}")
        else:
            lines.append(f"  {key:<25} {formatted:>10}  {target_str}")
    lines.append("=" * 50)
    return "\n".join(lines)
