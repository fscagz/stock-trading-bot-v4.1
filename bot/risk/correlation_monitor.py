"""
Correlation and factor crowding monitor.

Factor crowding occurs when many systematic strategies hold similar positions.
Popular factor tilts (momentum, quality) can unwind simultaneously when
crowded strategies de-risk together, amplifying drawdowns beyond what
factor models predict.

This module detects two related signals:
  1. Portfolio average pairwise correlation — high correlation means
     diversification is lower than expected, and stress scenarios will
     be more severe.
  2. Factor crowding score — cosine similarity between the portfolio's
     factor loadings and an external estimate of consensus factor
     positioning (e.g., top-quintile holdings from prior period).

Usage
-----
    from risk.correlation_monitor import CorrelationMonitor, CrowdingConfig

    monitor = CorrelationMonitor(config=CrowdingConfig())
    alert, details = monitor.check(
        weights=portfolio_weights,
        returns_panel=price_returns,
        as_of=rebalance_date,
    )
    if alert:
        print(f"CROWDING ALERT: {details['avg_correlation']:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CrowdingConfig:
    """
    Crowding and correlation alert thresholds.

    Attributes
    ----------
    avg_corr_alert : float
        Alert if portfolio-weighted average pairwise correlation exceeds this.
        Default 0.65 — at this level, the portfolio is moving like a single stock.
    corr_window_days : int
        Rolling window for realised correlation estimation (default 63 = ~3 months).
    min_stocks : int
        Minimum number of holdings required to compute a meaningful correlation.
    """
    avg_corr_alert: float = 0.65
    corr_window_days: int = 63
    min_stocks: int = 5


DEFAULT_CROWDING_CONFIG = CrowdingConfig()


# ---------------------------------------------------------------------------
# Correlation utilities
# ---------------------------------------------------------------------------

def compute_realized_correlation(
    returns_panel: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """
    Compute the rolling pairwise correlation matrix using the most recent
    `window` trading days.

    Parameters
    ----------
    returns_panel : pd.DataFrame
        Rows = dates, columns = tickers, values = daily returns.
    window : int
        Number of trading days to use.

    Returns
    -------
    pd.DataFrame — correlation matrix (tickers × tickers).
    """
    if returns_panel.empty or len(returns_panel) < 10:
        return pd.DataFrame()
    recent = returns_panel.iloc[-window:].dropna(axis=1, how="all")
    if recent.shape[1] < 2:
        return pd.DataFrame()
    return recent.corr()


def portfolio_avg_correlation(
    weights: Dict[str, float],
    returns_panel: pd.DataFrame,
    window: int = 63,
) -> float:
    """
    Compute the weighted average pairwise correlation of portfolio holdings.

    This is the portfolio-weighted mean of all off-diagonal correlation
    pairs. A value close to 1.0 means the holdings move together (no
    diversification benefit). A value near 0 means holdings are uncorrelated.

    Parameters
    ----------
    weights : dict {ticker: weight}
    returns_panel : pd.DataFrame (dates × tickers, daily returns)
    window : int

    Returns
    -------
    float — weighted average pairwise correlation (NaN if insufficient data).
    """
    held = [t for t in weights if t in returns_panel.columns and weights[t] > 0]
    if len(held) < 2:
        return float("nan")

    corr = compute_realized_correlation(returns_panel[held], window)
    if corr.empty:
        return float("nan")

    w = pd.Series({t: weights.get(t, 0.0) for t in corr.index})
    w = w / w.sum() if w.sum() > 0 else w

    # Weighted average of off-diagonal elements
    # avg_corr = sum_{i≠j} w_i * w_j * corr_ij / sum_{i≠j} w_i * w_j
    n = len(corr)
    if n < 2:
        return float("nan")

    numerator = 0.0
    denominator = 0.0
    tickers = corr.index.tolist()
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = tickers[i], tickers[j]
            wi, wj = float(w.get(ti, 0)), float(w.get(tj, 0))
            c = float(corr.loc[ti, tj])
            if not np.isnan(c):
                pair_w = wi * wj
                numerator += pair_w * c
                denominator += pair_w

    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def crowding_alert(avg_corr: float, threshold: float = 0.65) -> bool:
    """
    Return True if average portfolio correlation exceeds threshold.

    Parameters
    ----------
    avg_corr : float
    threshold : float

    Returns
    -------
    bool
    """
    if np.isnan(avg_corr):
        return False
    return avg_corr > threshold


# ---------------------------------------------------------------------------
# Factor crowding score
# ---------------------------------------------------------------------------

def factor_crowding_score(
    portfolio_factor_loadings: pd.Series,
    consensus_factor_loadings: pd.Series,
) -> float:
    """
    Compute cosine similarity between portfolio factor loadings and a
    consensus (benchmark) factor exposure vector.

    A score close to 1.0 means the portfolio's factor bets are identical
    to consensus — it is fully crowded into popular factors.
    A score near 0 means the portfolio has differentiated factor bets.

    Parameters
    ----------
    portfolio_factor_loadings : pd.Series
        Factor beta exposures for the portfolio (e.g. from FF5 regression).
        Index = factor names.
    consensus_factor_loadings : pd.Series
        Market consensus factor exposures (same index).

    Returns
    -------
    float — cosine similarity [-1, 1]. NaN if vectors are zero-length.
    """
    common = portfolio_factor_loadings.index.intersection(consensus_factor_loadings.index)
    if len(common) < 2:
        return float("nan")

    a = portfolio_factor_loadings.loc[common].values.astype(float)
    b = consensus_factor_loadings.loc[common].values.astype(float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return float("nan")

    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Monitor class
# ---------------------------------------------------------------------------

class CorrelationMonitor:
    """
    Monitors portfolio correlation and crowding on each rebalance.

    Parameters
    ----------
    config : CrowdingConfig
    """

    def __init__(self, config: CrowdingConfig = DEFAULT_CROWDING_CONFIG):
        self.config = config
        self._history: list = []

    def check(
        self,
        weights: Dict[str, float],
        returns_panel: pd.DataFrame,
        as_of=None,
        consensus_loadings: Optional[pd.Series] = None,
        portfolio_loadings: Optional[pd.Series] = None,
    ) -> Tuple[bool, Dict]:
        """
        Run correlation and crowding checks.

        Parameters
        ----------
        weights : dict {ticker: weight}
        returns_panel : pd.DataFrame (dates × tickers)
        as_of : date, optional (for logging)
        consensus_loadings : pd.Series, optional
            If provided along with portfolio_loadings, compute crowding score.
        portfolio_loadings : pd.Series, optional

        Returns
        -------
        (alert: bool, details: dict)
            alert is True if any threshold is exceeded.
        """
        n_held = len([t for t in weights if weights.get(t, 0) > 0])

        avg_corr = float("nan")
        if n_held >= self.config.min_stocks and not returns_panel.empty:
            avg_corr = portfolio_avg_correlation(
                weights, returns_panel, self.config.corr_window_days
            )

        corr_alert = crowding_alert(avg_corr, self.config.avg_corr_alert)

        crowd_score = float("nan")
        if consensus_loadings is not None and portfolio_loadings is not None:
            crowd_score = factor_crowding_score(portfolio_loadings, consensus_loadings)

        alert = corr_alert or (not np.isnan(crowd_score) and crowd_score > 0.90)

        details = {
            "as_of": as_of,
            "n_holdings": n_held,
            "avg_correlation": avg_corr,
            "corr_alert": corr_alert,
            "corr_threshold": self.config.avg_corr_alert,
            "crowding_score": crowd_score,
            "alert": alert,
        }

        self._history.append(details)
        return alert, details

    def history_df(self) -> pd.DataFrame:
        """Return all past check results as a DataFrame."""
        return pd.DataFrame(self._history)

    def latest_alert(self) -> bool:
        if not self._history:
            return False
        return bool(self._history[-1].get("alert", False))
