"""
Drawdown control — dynamic risk-off mechanism.

If the portfolio drawdown from its peak exceeds a threshold, gross exposure
is reduced automatically. This enforces the vision's hard limit of
≤ 25% maximum drawdown.

Regime ladder
-------------
  'normal'  : drawdown < warning_threshold (default 15%)
               → full exposure (no scaling)
  'warning' : warning_threshold ≤ drawdown < halt_threshold
               → 60% gross exposure (40% cash)
  'halt'    : drawdown ≥ halt_threshold (default 25%)
               → 20% gross exposure (80% cash). Hard stop.

Restoration: the regime does NOT automatically revert when the portfolio
recovers. The portfolio must recover to within restore_pct of the prior
peak before the regime steps down (e.g., warning → normal).

This prevents repeatedly oscillating in/out of risk-off around the threshold.

Usage
-----
    from risk.drawdown_control import DrawdownController, DrawdownConfig

    controller = DrawdownController(config=DrawdownConfig())
    regime = controller.update(nav_series)
    scaled_weights = controller.scale_weights(target_weights)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DrawdownConfig:
    """
    Drawdown control thresholds.

    Attributes
    ----------
    warning_threshold : float
        Drawdown at which exposure is reduced to warning_exposure.
    halt_threshold : float
        Drawdown at which exposure is reduced to halt_exposure.
        This is the hard stop from the vision (25%).
    warning_exposure : float
        Gross exposure multiplier in 'warning' regime (default 0.60).
    halt_exposure : float
        Gross exposure multiplier in 'halt' regime (default 0.20).
    restore_pct : float
        Portfolio must recover to within this fraction of prior peak
        before stepping down a regime (default 0.05 = within 5% of peak).
    """
    warning_threshold: float = 0.15
    halt_threshold: float = 0.25
    warning_exposure: float = 0.60
    halt_exposure: float = 0.20
    restore_pct: float = 0.05


DEFAULT_DRAWDOWN_CONFIG = DrawdownConfig()

REGIME_EXPOSURE = {
    "normal":  1.00,
    "warning": 0.60,
    "halt":    0.20,
}


# ---------------------------------------------------------------------------
# Stateless utilities
# ---------------------------------------------------------------------------

def current_drawdown(nav_series: pd.Series) -> float:
    """
    Compute the current drawdown from the rolling peak.

    Parameters
    ----------
    nav_series : pd.Series
        Portfolio NAV values (daily), DatetimeIndex.

    Returns
    -------
    float — current drawdown as a negative fraction (e.g. -0.12 = -12%).
            Returns 0.0 for empty or single-element series.
    """
    if len(nav_series) < 2:
        return 0.0
    rolling_max = nav_series.cummax()
    dd = (nav_series - rolling_max) / rolling_max
    return float(dd.iloc[-1])


def max_drawdown_from_series(nav_series: pd.Series) -> float:
    """
    Maximum drawdown over the full NAV series.

    Returns
    -------
    float — most negative drawdown (e.g. -0.35 = -35% max drawdown).
    """
    if len(nav_series) < 2:
        return 0.0
    rolling_max = nav_series.cummax()
    dd = (nav_series - rolling_max) / rolling_max
    return float(dd.min())


def drawdown_series(nav_series: pd.Series) -> pd.Series:
    """Return the full drawdown time series (all values ≤ 0)."""
    if nav_series.empty:
        return pd.Series(dtype=float)
    rolling_max = nav_series.cummax()
    return (nav_series - rolling_max) / rolling_max


def classify_drawdown_regime(
    drawdown: float,
    config: DrawdownConfig = DEFAULT_DRAWDOWN_CONFIG,
) -> str:
    """
    Classify a single drawdown value into a regime string.

    Parameters
    ----------
    drawdown : float — current drawdown (negative, e.g. -0.18)
    config : DrawdownConfig

    Returns
    -------
    str — 'normal', 'warning', or 'halt'
    """
    abs_dd = abs(drawdown)
    if abs_dd >= config.halt_threshold:
        return "halt"
    if abs_dd >= config.warning_threshold:
        return "warning"
    return "normal"


def scale_weights_for_regime(
    weights: Dict[str, float],
    regime: str,
    config: DrawdownConfig = DEFAULT_DRAWDOWN_CONFIG,
) -> Dict[str, float]:
    """
    Scale gross exposure according to the drawdown regime.

    Cash (unallocated portion) is implicit — weights will sum to less
    than 1, with the remainder held as cash.

    Parameters
    ----------
    weights : dict {ticker: weight}
        Full-exposure weights (should sum to 1 or less).
    regime : str — 'normal', 'warning', or 'halt'
    config : DrawdownConfig

    Returns
    -------
    dict {ticker: scaled_weight}
        Weights scaled by the regime's exposure multiplier.
        Sum will be < 1 in warning/halt regimes (cash = 1 - sum).
    """
    exposure = config.warning_exposure if regime == "warning" else (
        config.halt_exposure if regime == "halt" else 1.0
    )
    return {t: w * exposure for t, w in weights.items()}


# ---------------------------------------------------------------------------
# Stateful controller
# ---------------------------------------------------------------------------

@dataclass
class DrawdownState:
    """Persisted state for the DrawdownController."""
    peak_nav: float = 1.0
    current_regime: str = "normal"
    regime_entry_date: Optional[pd.Timestamp] = None
    regime_history: List[Tuple[pd.Timestamp, str]] = field(default_factory=list)


class DrawdownController:
    """
    Stateful drawdown controller that tracks regime and prevents thrashing.

    Call `update(nav_series)` after each rebalance to refresh the regime.
    Then call `scale_weights(target_weights)` to get exposure-adjusted weights.

    Parameters
    ----------
    config : DrawdownConfig
    """

    def __init__(self, config: DrawdownConfig = DEFAULT_DRAWDOWN_CONFIG):
        self.config = config
        self.state = DrawdownState()

    def update(self, nav_series: pd.Series) -> str:
        """
        Update regime based on the latest NAV series.

        Regime transitions:
          - Can worsen (normal → warning → halt) immediately when threshold crossed.
          - Can only improve (halt → warning → normal) when portfolio recovers
            to within `restore_pct` of the peak at the time the regime was entered.

        Parameters
        ----------
        nav_series : pd.Series
            Full portfolio NAV history (DatetimeIndex).

        Returns
        -------
        str — current regime after update.
        """
        if nav_series.empty:
            return self.state.current_regime

        current_nav = float(nav_series.iloc[-1])
        peak_nav = float(nav_series.cummax().iloc[-1])
        self.state.peak_nav = peak_nav

        dd = current_drawdown(nav_series)
        new_regime = classify_drawdown_regime(dd, self.config)
        ts = nav_series.index[-1] if hasattr(nav_series.index[-1], 'date') else pd.Timestamp.now()

        current = self.state.current_regime
        regime_order = ["normal", "warning", "halt"]

        worsening = regime_order.index(new_regime) > regime_order.index(current)
        improving = regime_order.index(new_regime) < regime_order.index(current)

        if worsening:
            self.state.current_regime = new_regime
            self.state.regime_entry_date = ts
            self.state.regime_history.append((ts, new_regime))

        elif improving:
            # Only recover if current NAV is within restore_pct of peak
            recovery_threshold = peak_nav * (1 - self.config.restore_pct)
            if current_nav >= recovery_threshold:
                self.state.current_regime = new_regime
                self.state.regime_history.append((ts, new_regime))

        return self.state.current_regime

    def scale_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply current regime's exposure scaling to target weights.

        Returns scaled weights (sum ≤ 1; remainder is implicit cash).
        """
        return scale_weights_for_regime(weights, self.state.current_regime, self.config)

    @property
    def current_regime(self) -> str:
        return self.state.current_regime

    def is_halted(self) -> bool:
        return self.state.current_regime == "halt"

    def is_in_warning(self) -> bool:
        return self.state.current_regime == "warning"

    def exposure(self) -> float:
        """Return current gross exposure multiplier (1.0, 0.6, or 0.2)."""
        return REGIME_EXPOSURE[self.state.current_regime]

    def summary(self) -> str:
        dd_pct = (1 - self.state.peak_nav / max(self.state.peak_nav, 1e-9))
        return (
            f"DrawdownController: regime={self.state.current_regime!r}  "
            f"peak_nav={self.state.peak_nav:.4f}  "
            f"exposure={self.exposure():.0%}  "
            f"n_regime_changes={len(self.state.regime_history)}"
        )
