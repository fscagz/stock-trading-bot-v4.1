"""
Top-level risk model.

Single interface used by the backtest engine and live pipeline.
Combines position sizing, hard constraints, drawdown control,
and crowding monitoring into one `.run()` call.

Input  : raw composite scores + vol estimates + portfolio state
Output : constrained, sized, drawdown-scaled weight vector + risk report

Usage
-----
    from risk.risk_model import RiskModel, RiskConfig

    model = RiskModel(RiskConfig())

    weights, report = model.run(
        scores=composite_scores,          # pd.Series — selected tickers
        vol_estimates=ewma_vols,          # pd.Series — annualised vol per ticker
        prev_weights=current_portfolio,   # dict — weights before rebalance
        nav_series=portfolio_nav,         # pd.Series — daily NAV history
        sector_map=sector_lookup,         # dict {ticker: sector}
        returns_panel=daily_returns_df,   # pd.DataFrame — for correlation check
    )

    if report["drawdown_regime"] == "halt":
        print("DRAWDOWN HALT — moving to near-cash")

    if report["crowding_alert"]:
        print("CROWDING ALERT — reduce factor concentration")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd

from risk.constraints import (
    ConstraintConfig,
    DEFAULT_CONSTRAINTS,
    apply_all_constraints,
    compute_one_way_turnover,
)
from risk.correlation_monitor import (
    CorrelationMonitor,
    CrowdingConfig,
    DEFAULT_CROWDING_CONFIG,
)
from risk.drawdown_control import (
    DrawdownConfig,
    DrawdownController,
    DEFAULT_DRAWDOWN_CONFIG,
    classify_drawdown_regime,
    current_drawdown,
)
from risk.position_sizing import (
    SizingConfig,
    DEFAULT_SIZING,
    size_positions,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RiskConfig:
    """
    Unified risk model configuration.

    Attributes
    ----------
    sizing : SizingConfig
        Position sizing method and vol target.
    constraints : ConstraintConfig
        Hard constraints (position caps, sector caps, turnover).
    drawdown : DrawdownConfig
        Drawdown regime thresholds and exposure scaling.
    crowding : CrowdingConfig
        Correlation and crowding alert thresholds.
    run_crowding_check : bool
        If False, skip correlation monitor (faster for parameter sweeps).
    """
    sizing: SizingConfig = None
    constraints: ConstraintConfig = None
    drawdown: DrawdownConfig = None
    crowding: CrowdingConfig = None
    run_crowding_check: bool = True

    def __post_init__(self):
        if self.sizing is None:
            self.sizing = SizingConfig()
        if self.constraints is None:
            self.constraints = ConstraintConfig()
        if self.drawdown is None:
            self.drawdown = DrawdownConfig()
        if self.crowding is None:
            self.crowding = CrowdingConfig()


DEFAULT_RISK_CONFIG = RiskConfig()


# ---------------------------------------------------------------------------
# Risk Model
# ---------------------------------------------------------------------------

class RiskModel:
    """
    Stateful risk model.

    State: DrawdownController (tracks regime across rebalances).
    The CorrelationMonitor accumulates history but does not affect weights
    beyond raising alerts.

    Parameters
    ----------
    config : RiskConfig
    """

    def __init__(self, config: RiskConfig = DEFAULT_RISK_CONFIG):
        self.config = config
        self._drawdown_controller = DrawdownController(config.drawdown)
        self._correlation_monitor = CorrelationMonitor(config.crowding)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        scores: pd.Series,
        vol_estimates: Optional[pd.Series] = None,
        prev_weights: Optional[Dict[str, float]] = None,
        nav_series: Optional[pd.Series] = None,
        sector_map: Optional[Dict[str, str]] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
        returns_panel: Optional[pd.DataFrame] = None,
        as_of=None,
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Full risk pipeline: size → constrain → drawdown scale.

        Parameters
        ----------
        scores : pd.Series
            Composite scores for selected tickers (higher = better).
            Only non-NaN tickers are sized.
        vol_estimates : pd.Series, optional
            Annualised EWMA vol per ticker. Required for vol_target/risk_parity.
        prev_weights : dict, optional
            Current portfolio weights. Used for turnover constraint and
            drawdown controller.
        nav_series : pd.Series, optional
            Portfolio daily NAV. Used to update drawdown regime.
        sector_map : dict {ticker: sector_name}, optional
            Used for sector caps.
        cov_matrix : pd.DataFrame, optional
            Used for risk_parity sizing.
        returns_panel : pd.DataFrame, optional
            Daily returns (dates × tickers). Used for crowding check.
        as_of : date-like, optional
            Rebalance date (for logging).

        Returns
        -------
        (weights: dict {ticker: weight}, risk_report: dict)

        risk_report keys:
            sized_weights       : weights after sizing (pre-constraint)
            constrained_weights : weights after constraints (pre-drawdown)
            final_weights       : weights after all adjustments
            drawdown_regime     : 'normal' | 'warning' | 'halt'
            current_drawdown    : float (e.g. -0.12 = -12%)
            gross_exposure      : float (1.0, 0.6, or 0.2)
            one_way_turnover    : float
            crowding_alert      : bool
            avg_correlation     : float
            n_positions         : int
        """
        prev = prev_weights or {}

        # 1. Update drawdown regime
        dd = 0.0
        regime = "normal"
        if nav_series is not None and not nav_series.empty:
            regime = self._drawdown_controller.update(nav_series)
            dd = current_drawdown(nav_series)

        # 2. Position sizing
        sized = size_positions(
            scores=scores,
            config=self.config.sizing,
            vol_estimates=vol_estimates,
            cov_matrix=cov_matrix,
        )

        # 3. Hard constraints
        constrained = apply_all_constraints(
            target_weights=sized,
            prev_weights=prev if prev else None,
            sector_map=sector_map,
            config=self.config.constraints,
        )

        # 4. Drawdown exposure scaling
        final = self._drawdown_controller.scale_weights(constrained)

        # 5. Correlation / crowding check
        crowding_triggered = False
        avg_corr = float("nan")
        if self.config.run_crowding_check and returns_panel is not None:
            alert, details = self._correlation_monitor.check(
                weights=constrained,
                returns_panel=returns_panel,
                as_of=as_of,
            )
            crowding_triggered = alert
            avg_corr = details.get("avg_correlation", float("nan"))

        # 6. Compute turnover
        to = compute_one_way_turnover(prev, final)

        report = {
            "sized_weights": sized,
            "constrained_weights": constrained,
            "final_weights": final,
            "drawdown_regime": regime,
            "current_drawdown": dd,
            "gross_exposure": self._drawdown_controller.exposure(),
            "one_way_turnover": to,
            "crowding_alert": crowding_triggered,
            "avg_correlation": avg_corr,
            "n_positions": len([w for w in final.values() if w > 1e-4]),
        }

        return final, report

    # ------------------------------------------------------------------
    # Individual step access (for testing / inspection)
    # ------------------------------------------------------------------

    def size_only(
        self,
        scores: pd.Series,
        vol_estimates: Optional[pd.Series] = None,
        cov_matrix: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """Sizing step only (no constraints, no drawdown)."""
        return size_positions(scores, self.config.sizing, vol_estimates, cov_matrix)

    def constrain_only(
        self,
        weights: Dict[str, float],
        prev_weights: Optional[Dict[str, float]] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """Constraint step only."""
        return apply_all_constraints(weights, prev_weights, sector_map, self.config.constraints)

    def drawdown_regime(self, nav_series: pd.Series) -> str:
        """Return current drawdown regime without updating state."""
        dd = current_drawdown(nav_series)
        return classify_drawdown_regime(dd, self.config.drawdown)

    @property
    def current_regime(self) -> str:
        return self._drawdown_controller.state.current_regime

    @property
    def peak_nav(self) -> float:
        return self._drawdown_controller.state.peak_nav

    def reset(self) -> None:
        """Reset stateful components (use between independent backtests)."""
        self._drawdown_controller = DrawdownController(self.config.drawdown)
        self._correlation_monitor = CorrelationMonitor(self.config.crowding)
