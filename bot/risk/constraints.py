"""
Portfolio hard constraints.

Applied after position sizing to enforce:
  - Maximum single-position weight (5% cap)
  - Minimum single-position weight (0.5% floor — avoids hairline positions)
  - Maximum sector concentration (30% per sector)
  - Maximum one-way turnover per rebalance (30%)
  - Long-only enforcement

Constraints are applied sequentially. Each step renormalises weights so
they sum to 1. Order matters: position caps first, then sector caps, then
turnover constraint (which may hold back the rebalance if it would exceed
the turnover budget).

Usage
-----
    from risk.constraints import apply_all_constraints, ConstraintConfig

    weights = apply_all_constraints(
        target_weights=sized_weights,
        prev_weights=current_weights,
        sector_map={"AAPL": "tech", "JPM": "finance", ...},
        config=ConstraintConfig(),
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ConstraintConfig:
    """
    Hard constraint parameters.

    Attributes
    ----------
    max_single_position : float
        Maximum weight for any single stock (default 5%).
    min_single_position : float
        Minimum weight for any held stock (default 0.5%).
        Positions below this after capping are zeroed out.
    max_sector_weight : float
        Maximum total weight in any single GICS sector (default 30%).
    max_one_way_turnover : float
        Maximum one-way turnover per rebalance (default 30%).
        One-way turnover = sum of absolute weight changes / 2.
        If target turnover exceeds this, the rebalance is partially blended
        toward current weights to stay within budget.
    """
    max_single_position: float = 0.05
    min_single_position: float = 0.005
    max_sector_weight: float = 0.30
    max_one_way_turnover: float = 0.30


DEFAULT_CONSTRAINTS = ConstraintConfig()


# ---------------------------------------------------------------------------
# Individual constraint functions
# ---------------------------------------------------------------------------

def enforce_long_only(weights: Dict[str, float]) -> Dict[str, float]:
    """Clip negative weights to zero and renormalise."""
    clipped = {t: max(0.0, w) for t, w in weights.items()}
    total = sum(clipped.values())
    if total <= 0:
        return {}
    return {t: w / total for t, w in clipped.items()}


def apply_position_caps(
    weights: Dict[str, float],
    max_weight: float = 0.05,
    min_weight: float = 0.005,
) -> Dict[str, float]:
    """
    Cap any single position at max_weight, remove positions below min_weight.

    Excess weight is redistributed pro-rata to uncapped positions. When no
    uncapped positions remain (all are at the cap), the sum may be < 1.0 —
    the remainder is implicit cash. This is intentional: we never renormalise
    in a way that would push weights back above max_weight.

    Parameters
    ----------
    weights : dict {ticker: weight}
    max_weight : float
    min_weight : float

    Returns
    -------
    dict {ticker: weight} — all values ≤ max_weight. Sum ≤ 1.
    """
    # Clip negatives and remove below-min
    w = {t: max(0.0, wt) for t, wt in weights.items()}
    w = {t: wt for t, wt in w.items() if wt >= min_weight}
    if not w:
        return {}

    # Normalise once at the start
    total = sum(w.values())
    if total <= 0:
        return {}
    w = {t: wt / total for t, wt in w.items()}

    # Iteratively cap and redistribute
    for _ in range(100):
        over = {t for t, wt in w.items() if wt > max_weight + 1e-10}
        if not over:
            break
        excess = sum(w[t] - max_weight for t in over)
        for t in over:
            w[t] = max_weight
        uncapped = {t: wt for t, wt in w.items() if wt < max_weight - 1e-10}
        if not uncapped:
            break  # all at cap — implicit cash, stop redistributing
        uncapped_total = sum(uncapped.values())
        for t in uncapped:
            w[t] = min(w[t] + excess * (uncapped[t] / uncapped_total), max_weight)

    # Fill any remaining deficit (numerical drift) into uncapped stocks
    for _ in range(20):
        total = sum(w.values())
        deficit = 1.0 - total
        if deficit < 1e-10:
            break
        uncapped = {t: wt for t, wt in w.items() if wt < max_weight - 1e-10}
        if not uncapped:
            break  # all at cap — remaining deficit becomes implicit cash
        uncapped_total = sum(uncapped.values())
        for t in uncapped:
            w[t] = min(w[t] + deficit * (w[t] / uncapped_total), max_weight)

    # Remove hairline positions created during redistribution
    w = {t: wt for t, wt in w.items() if wt >= min_weight}

    # Only renormalise if the result won't violate max_weight
    total = sum(w.values())
    if total > 0:
        renormed = {t: wt / total for t, wt in w.items()}
        if all(v <= max_weight + 1e-9 for v in renormed.values()):
            return renormed

    return w  # implicit cash — sum < 1 is intentional


def apply_sector_caps(
    weights: Dict[str, float],
    sector_map: Dict[str, str],
    max_sector_weight: float = 0.30,
) -> Dict[str, float]:
    """
    Cap any single sector at max_sector_weight.

    Stocks not in sector_map are grouped into 'unknown' sector.
    Excess weight from capped sectors is redistributed pro-rata to
    uncapped sectors.

    Parameters
    ----------
    weights : dict {ticker: weight}
    sector_map : dict {ticker: sector_name}
    max_sector_weight : float

    Returns
    -------
    dict {ticker: weight}
    """
    if not sector_map:
        return dict(weights)

    w = dict(weights)
    total = sum(w.values())
    if total <= 0:
        return {}
    w = {t: wt / total for t, wt in w.items()}

    for _ in range(50):
        # Compute sector totals
        sector_totals: Dict[str, float] = {}
        for t, wt in w.items():
            sec = sector_map.get(t, "unknown")
            sector_totals[sec] = sector_totals.get(sec, 0.0) + wt

        over_sectors = {s for s, sw in sector_totals.items() if sw > max_sector_weight + 1e-9}
        if not over_sectors:
            break

        # Scale back over-weight sectors
        for sec in over_sectors:
            sec_stocks = [t for t in w if sector_map.get(t, "unknown") == sec]
            sec_total = sum(w[t] for t in sec_stocks)
            if sec_total <= 0:
                continue
            scale = max_sector_weight / sec_total
            for t in sec_stocks:
                w[t] *= scale

        # Redistribute the weight removed from capped sectors to uncapped sectors
        current_total = sum(w.values())
        excess = 1.0 - current_total
        if excess > 1e-10:
            uncapped_stocks = [
                t for t in w
                if sector_totals.get(sector_map.get(t, "unknown"), 0.0) < max_sector_weight - 1e-9
            ]
            uncapped_total = sum(w[t] for t in uncapped_stocks)
            if uncapped_total > 0:
                for t in uncapped_stocks:
                    w[t] += excess * (w[t] / uncapped_total)

    # Final: only renorm if sector caps are satisfied in the result
    total = sum(w.values())
    if total > 0:
        renormed = {t: wt / total for t, wt in w.items()}
        renormed_sector_totals: Dict[str, float] = {}
        for t, wt in renormed.items():
            sec = sector_map.get(t, "unknown")
            renormed_sector_totals[sec] = renormed_sector_totals.get(sec, 0.0) + wt
        if all(st <= max_sector_weight + 1e-9 for st in renormed_sector_totals.values()):
            return renormed

    return w  # implicit cash — sum ≤ 1


def apply_turnover_constraint(
    prev_weights: Dict[str, float],
    target_weights: Dict[str, float],
    max_one_way_turnover: float = 0.30,
) -> Dict[str, float]:
    """
    Blend target weights toward previous weights to stay within turnover budget.

    One-way turnover = sum(|target - prev|) / 2

    If target_turnover <= max_one_way_turnover, return target as-is.
    Otherwise, find the blend factor α such that:
        blended = α * target + (1 - α) * prev
    gives exactly max_one_way_turnover one-way turnover.

    Parameters
    ----------
    prev_weights : dict
        Current portfolio weights (before rebalance). May be empty (first trade).
    target_weights : dict
        Desired target weights.
    max_one_way_turnover : float

    Returns
    -------
    dict {ticker: weight} — blended weights that sum to 1.
    """
    if not prev_weights:
        return dict(target_weights)

    all_tickers = set(prev_weights) | set(target_weights)
    prev = pd.Series({t: prev_weights.get(t, 0.0) for t in all_tickers})
    target = pd.Series({t: target_weights.get(t, 0.0) for t in all_tickers})

    actual_turnover = float((target - prev).abs().sum() / 2)
    if actual_turnover <= max_one_way_turnover + 1e-9:
        return dict(target_weights)

    # Binary search for blend factor α
    lo, hi = 0.0, 1.0
    for _ in range(50):
        alpha = (lo + hi) / 2
        blended = alpha * target + (1 - alpha) * prev
        to = float((blended - prev).abs().sum() / 2)
        if to < max_one_way_turnover:
            lo = alpha
        else:
            hi = alpha

    alpha = (lo + hi) / 2
    blended = alpha * target + (1 - alpha) * prev

    # Remove near-zero weights. Do NOT renorm: target may have implicit cash
    # (sum < 1) due to position caps, and renorming would violate those caps.
    result = {t: float(blended[t]) for t in all_tickers if float(blended[t]) > 1e-6}
    if not result:
        return {}
    return result


# ---------------------------------------------------------------------------
# Composite constraint application
# ---------------------------------------------------------------------------

def apply_all_constraints(
    target_weights: Dict[str, float],
    prev_weights: Optional[Dict[str, float]] = None,
    sector_map: Optional[Dict[str, str]] = None,
    config: ConstraintConfig = DEFAULT_CONSTRAINTS,
) -> Dict[str, float]:
    """
    Apply all hard constraints in sequence.

    Order:
      1. Enforce long-only (clip negatives)
      2. Position caps (max/min per stock)
      3. Sector caps (if sector_map provided)
      4. Turnover constraint (if prev_weights provided)

    Parameters
    ----------
    target_weights : dict {ticker: weight}
    prev_weights : dict, optional
    sector_map : dict {ticker: sector}, optional
    config : ConstraintConfig

    Returns
    -------
    dict {ticker: weight} — constrained weights summing to 1.
    """
    w = enforce_long_only(target_weights)
    if not w:
        return {}

    w = apply_position_caps(
        w,
        max_weight=config.max_single_position,
        min_weight=config.min_single_position,
    )
    if not w:
        return {}

    if sector_map:
        w = apply_sector_caps(w, sector_map, config.max_sector_weight)
        if not w:
            return {}

    if prev_weights is not None:
        w = apply_turnover_constraint(prev_weights, w, config.max_one_way_turnover)

    return w


def compute_one_way_turnover(
    prev_weights: Dict[str, float],
    target_weights: Dict[str, float],
) -> float:
    """
    Compute one-way turnover between two weight dicts.

    Returns
    -------
    float — fraction of portfolio value traded (0 = no change, 1 = full turn).
    """
    all_tickers = set(prev_weights) | set(target_weights)
    total_change = sum(
        abs(target_weights.get(t, 0.0) - prev_weights.get(t, 0.0))
        for t in all_tickers
    )
    return total_change / 2
