"""
Transaction cost and slippage model.

Two components:
  1. Bid-ask spread  — fixed percentage of trade value, per side.
  2. Market impact   — optional square-root model proportional to trade size
                       relative to average daily volume (ADV).

For large-cap US equities, a conservative assumption is:
  - Spread: 5–10 bps per side (use 10 bps as default for conservatism)
  - Impact: negligible for portfolios < $10M AUM; relevant for larger books

The cost model is parameterised so it can be changed without touching
engine logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class CostModel:
    """
    Parameters for estimating one-way transaction costs.

    Attributes
    ----------
    spread_bps : float
        Half-spread per side in basis points (default 10 bps = 0.10%).
        Applied to every trade regardless of size.
    use_market_impact : bool
        If True, add a square-root market impact term.
    impact_coefficient : float
        Scales the market impact term. The square-root model is:
          impact = impact_coefficient * sqrt(trade_fraction_of_adv)
        where trade_fraction_of_adv = trade_value / (adv_shares * price).
        A typical value is 0.1 (Almgren et al.).
    min_adv_for_impact : float
        Minimum ADV (shares) required to apply impact model. Below this,
        assume the impact is already covered by the spread.
    """
    spread_bps: float = 10.0
    use_market_impact: bool = False
    impact_coefficient: float = 0.1
    min_adv_for_impact: float = 100_000.0

    @property
    def spread_pct(self) -> float:
        """One-way spread as a decimal (e.g. 10 bps → 0.001)."""
        return self.spread_bps / 10_000.0


DEFAULT_COST_MODEL = CostModel(spread_bps=10.0, use_market_impact=False)


def estimate_one_way_cost(
    trade_value: float,
    price: float,
    adv_shares: Optional[float] = None,
    model: CostModel = DEFAULT_COST_MODEL,
) -> float:
    """
    Estimate one-way transaction cost for a single trade.

    Parameters
    ----------
    trade_value : float
        Absolute dollar value of the trade (always positive).
    price : float
        Current price per share.
    adv_shares : float, optional
        Average daily volume in shares. Required for market impact.
    model : CostModel

    Returns
    -------
    float
        Estimated one-way cost in dollars.
    """
    if trade_value <= 0:
        return 0.0

    # Spread cost
    cost = trade_value * model.spread_pct

    # Market impact (square-root model)
    if model.use_market_impact and adv_shares is not None and adv_shares >= model.min_adv_for_impact:
        adv_value = adv_shares * price
        if adv_value > 0:
            trade_fraction = trade_value / adv_value
            impact_pct = model.impact_coefficient * (trade_fraction ** 0.5)
            cost += trade_value * impact_pct

    return cost


def compute_rebalance_costs(
    prev_weights: Dict[str, float],
    target_weights: Dict[str, float],
    prices: Dict[str, float],
    portfolio_value: float,
    model: CostModel = DEFAULT_COST_MODEL,
    adv_shares: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute total transaction costs for a rebalance.

    Parameters
    ----------
    prev_weights : dict {ticker: weight}
        Current portfolio weights (before rebalance). Sum ≈ 1.
    target_weights : dict {ticker: weight}
        Target portfolio weights (after rebalance). Sum ≈ 1.
    prices : dict {ticker: price}
        Current prices at the rebalance date.
    portfolio_value : float
        Current portfolio NAV in dollars.
    model : CostModel
    adv_shares : dict, optional
        {ticker: ADV in shares} for market impact calculation.

    Returns
    -------
    (total_cost_dollars, per_ticker_cost_dict)
        total_cost_dollars : float — total cost to deduct from NAV
        per_ticker_cost_dict : dict {ticker: cost_dollars}
    """
    # Collect all tickers involved in any trade
    all_tickers = set(prev_weights) | set(target_weights)
    per_ticker: Dict[str, float] = {}
    total = 0.0

    for ticker in all_tickers:
        prev_w = prev_weights.get(ticker, 0.0)
        tgt_w = target_weights.get(ticker, 0.0)
        delta_w = abs(tgt_w - prev_w)
        if delta_w < 1e-6:
            continue  # no meaningful trade

        trade_value = delta_w * portfolio_value
        price = prices.get(ticker, 0.0)
        if price <= 0:
            continue

        adv = adv_shares.get(ticker) if adv_shares else None
        cost = estimate_one_way_cost(trade_value, price, adv, model)
        per_ticker[ticker] = cost
        total += cost

    return total, per_ticker


def should_trade(
    prev_weight: float,
    target_weight: float,
    price: float,
    portfolio_value: float,
    model: CostModel = DEFAULT_COST_MODEL,
    min_net_alpha_pct: float = 0.001,
) -> bool:
    """
    Decide whether a trade is worth executing given its cost.

    Skip if expected alpha improvement (from weight change) net of transaction
    cost is below min_net_alpha_pct. This implements the transaction-cost-aware
    rebalancing described in the vision.

    Parameters
    ----------
    prev_weight, target_weight : float
    price : float
    portfolio_value : float
    model : CostModel
    min_net_alpha_pct : float
        Minimum expected net alpha improvement to justify the trade (default 0.1%).

    Returns
    -------
    bool
        True if the trade should be executed.
    """
    delta_w = abs(target_weight - prev_weight)
    if delta_w < 1e-6:
        return False
    trade_value = delta_w * portfolio_value
    cost = estimate_one_way_cost(trade_value, price, model=model)
    cost_pct = cost / portfolio_value if portfolio_value > 0 else 0.0
    return cost_pct < min_net_alpha_pct
