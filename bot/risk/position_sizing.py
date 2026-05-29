"""
Position sizing methods.

Translates composite scores into initial position weights that target
a specific portfolio-level risk budget.

Methods
-------
equal_weight        — all selected stocks equal weight (baseline)
score_proportional  — weight ∝ composite score rank
vol_target_weight   — inverse-vol weighting scaled to hit target_vol
risk_parity         — equal risk contribution (solves for weights numerically)

The vision mandates a 15% annualised volatility target. After sizing, weights
are passed to constraints.py for hard caps.

Usage
-----
    from risk.position_sizing import vol_target_weight, SizingConfig

    weights = vol_target_weight(
        scores=composite_scores,
        vol_estimates=ewma_vols,
        config=SizingConfig(target_vol=0.15, max_single=0.05),
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SizingConfig:
    """
    Configuration for position sizing.

    Attributes
    ----------
    target_vol : float
        Annualised portfolio volatility target (default 15%).
    max_single : float
        Hard cap on any single position before normalisation.
        Applied inside the sizing functions as a pre-normalisation guard.
        (Hard post-normalisation cap is in constraints.py.)
    min_single : float
        Minimum weight for any selected stock (avoids hairline positions).
    method : str
        'equal' | 'score' | 'vol_target' | 'risk_parity'
    """
    target_vol: float = 0.15
    max_single: float = 0.05
    min_single: float = 0.005
    method: str = "vol_target"


DEFAULT_SIZING = SizingConfig()


# ---------------------------------------------------------------------------
# EWMA volatility estimator
# ---------------------------------------------------------------------------

def ewma_vol(
    returns: pd.Series,
    lam: float = 0.94,
    min_periods: int = 20,
) -> float:
    """
    Compute annualised EWMA volatility for a single stock.

    Parameters
    ----------
    returns : pd.Series
        Daily returns (most recent last).
    lam : float
        EWMA decay factor (RiskMetrics default 0.94).
    min_periods : int
        Minimum observations required; returns NaN if insufficient.

    Returns
    -------
    float — annualised volatility.
    """
    r = returns.dropna()
    if len(r) < min_periods:
        return float("nan")

    # Compute variance iteratively (most-recent-last ordering)
    var = float(r.iloc[0] ** 2)
    for ret in r.iloc[1:]:
        var = lam * var + (1 - lam) * float(ret) ** 2

    return float(np.sqrt(var * 252))


def compute_vol_estimates(
    price_panel: Dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    lookback_days: int = 126,
    lam: float = 0.94,
) -> pd.Series:
    """
    Compute EWMA volatility for each ticker as of a given date.

    Parameters
    ----------
    price_panel : dict {ticker: daily_df with 'close' column}
    as_of : pd.Timestamp
        Point-in-time cutoff — no data after this date used.
    lookback_days : int
        Number of trading days to include in the vol estimate.
    lam : float
        EWMA decay factor.

    Returns
    -------
    pd.Series {ticker: annualised_vol}
    """
    vols = {}
    for ticker, df in price_panel.items():
        if "close" not in df.columns:
            continue
        prices = df["close"].loc[df.index <= as_of]
        if len(prices) < 2:
            continue
        rets = prices.pct_change().dropna().iloc[-lookback_days:]
        v = ewma_vol(rets, lam=lam)
        if not np.isnan(v) and v > 0:
            vols[ticker] = v
    return pd.Series(vols)


# ---------------------------------------------------------------------------
# Sizing methods
# ---------------------------------------------------------------------------

def equal_weight(tickers: List[str]) -> Dict[str, float]:
    """Equal weight across all tickers. Sums to 1."""
    if not tickers:
        return {}
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}


def score_proportional(
    scores: pd.Series,
    config: SizingConfig = DEFAULT_SIZING,
) -> Dict[str, float]:
    """
    Weight proportional to composite score rank (not raw score).

    Rank-based to reduce outlier sensitivity.
    Higher rank (better score) → higher weight.
    """
    valid = scores.dropna()
    if valid.empty:
        return {}
    ranks = valid.rank(ascending=True)  # rank 1 = worst
    total = ranks.sum()
    weights = {t: float(ranks[t] / total) for t in valid.index}
    # Apply pre-cap
    weights = {t: min(w, config.max_single) for t, w in weights.items()}
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {t: w / total for t, w in weights.items()}


def vol_target_weight(
    scores: pd.Series,
    vol_estimates: pd.Series,
    config: SizingConfig = DEFAULT_SIZING,
) -> Dict[str, float]:
    """
    Inverse-volatility weighting scaled to the portfolio volatility target.

    For each selected stock:
        raw_weight[i] = (target_vol / vol[i])

    Then normalise so weights sum to 1. The resulting portfolio (under
    the diagonal covariance assumption) will have vol ≈ target_vol.

    Stocks with missing vol estimates are excluded.

    Parameters
    ----------
    scores : pd.Series
        Selected tickers (non-NaN). Only tickers present here are sized.
    vol_estimates : pd.Series
        Annualised EWMA vol per ticker.
    config : SizingConfig

    Returns
    -------
    dict {ticker: weight}
    """
    valid_tickers = scores.dropna().index
    raw = {}
    for t in valid_tickers:
        if t not in vol_estimates or np.isnan(vol_estimates[t]) or vol_estimates[t] <= 0:
            continue
        raw[t] = config.target_vol / vol_estimates[t]

    if not raw:
        # Fallback: equal weight if no vol estimates available
        return equal_weight(list(valid_tickers))

    total = sum(raw.values())
    if total <= 0:
        return {}

    # Normalise to proportional weights
    weights = {t: w / total for t, w in raw.items()}

    # Apply cap and min (iterative redistribution, same logic as apply_position_caps)
    from risk.constraints import apply_position_caps
    return apply_position_caps(weights, max_weight=config.max_single, min_weight=config.min_single)


def risk_parity(
    scores: pd.Series,
    cov_matrix: pd.DataFrame,
    config: SizingConfig = DEFAULT_SIZING,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> Dict[str, float]:
    """
    Risk parity (equal risk contribution) weights.

    Each stock contributes equally to total portfolio variance.
    Uses the iterative (Maillard et al.) algorithm.

    Parameters
    ----------
    scores : pd.Series
        Selected tickers (determines universe; values not used for sizing).
    cov_matrix : pd.DataFrame
        Annualised covariance matrix. Must include all selected tickers.
    config : SizingConfig
    max_iter : int
        Maximum iterations for the solver.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict {ticker: weight}

    Notes
    -----
    Falls back to inverse-vol (diagonal of cov_matrix) if cov_matrix is
    ill-conditioned or tickers are missing.
    """
    tickers = [t for t in scores.dropna().index if t in cov_matrix.index]
    if not tickers:
        return {}

    n = len(tickers)
    Sigma = cov_matrix.loc[tickers, tickers].values.astype(float)

    # Regularise to ensure positive-definiteness
    Sigma += np.eye(n) * 1e-8

    # Iterative equal risk contribution (ERC) solver
    w = np.ones(n) / n  # start at equal weight
    for _ in range(max_iter):
        port_var = w @ Sigma @ w
        if port_var <= 0:
            break
        # Marginal risk contributions
        mrc = Sigma @ w
        # Risk contributions
        rc = w * mrc / port_var
        # Update: scale each weight so RC converges to 1/n
        w_new = w * (1 / n) / rc
        w_new = np.maximum(w_new, 1e-8)
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    # Apply pre-cap and min_single
    weights = {t: float(w[i]) for i, t in enumerate(tickers)}
    weights = {t: min(wt, config.max_single) for t, wt in weights.items()}
    weights = {t: wt for t, wt in weights.items() if wt >= config.min_single}
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {t: wt / total for t, wt in weights.items()}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def size_positions(
    scores: pd.Series,
    config: SizingConfig = DEFAULT_SIZING,
    vol_estimates: Optional[pd.Series] = None,
    cov_matrix: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Route to the correct sizing method based on config.method.

    Parameters
    ----------
    scores : pd.Series
        Selected tickers with composite scores.
    config : SizingConfig
    vol_estimates : pd.Series, optional
        Required for 'vol_target' and 'risk_parity' (diagonal fallback).
    cov_matrix : pd.DataFrame, optional
        Required for 'risk_parity'.

    Returns
    -------
    dict {ticker: weight}
    """
    if config.method == "equal":
        return equal_weight(list(scores.dropna().index))

    if config.method == "score":
        return score_proportional(scores, config)

    if config.method == "vol_target":
        if vol_estimates is None:
            return equal_weight(list(scores.dropna().index))
        return vol_target_weight(scores, vol_estimates, config)

    if config.method == "risk_parity":
        if cov_matrix is not None:
            return risk_parity(scores, cov_matrix, config)
        if vol_estimates is not None:
            # Diagonal cov fallback
            diag_cov = pd.DataFrame(
                np.diag(vol_estimates.reindex(scores.dropna().index).fillna(0.20) ** 2),
                index=scores.dropna().index,
                columns=scores.dropna().index,
            )
            return risk_parity(scores, diag_cov, config)
        return equal_weight(list(scores.dropna().index))

    raise ValueError(f"Unknown sizing method: {config.method!r}")
