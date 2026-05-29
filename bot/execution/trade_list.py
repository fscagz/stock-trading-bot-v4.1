"""
Trade list generation and cost filtering.

Converts target weights into an executable trade list with estimated costs,
then filters out uneconomical trades.

Steps
-----
1. compute delta_w for each ticker (target - current)
2. Estimate transaction cost for each trade
3. Filter trades where cost > min_net_alpha_pct of NAV
4. Return a DataFrame with trade details for execution

Usage
-----
    from execution.trade_list import generate_trade_list, filter_trades_by_cost

    trades = generate_trade_list(
        prev_weights=current_weights,
        target_weights=target_weights,
        prices=price_dict,
        nav=portfolio_value,
    )

    filtered = filter_trades_by_cost(
        trades,
        min_net_alpha_pct=0.001,  # skip if cost > 0.1% of NAV
    )

    for _, trade in filtered.iterrows():
        print(f"{trade['ticker']}: {trade['side']} {trade['shares']} @ ${trade['price']}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtest.costs import CostModel, DEFAULT_COST_MODEL, estimate_one_way_cost


# ---------------------------------------------------------------------------
# Trade data structure
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """A single trade to execute."""
    ticker: str
    side: str              # "buy" or "sell"
    shares: float
    price: float
    notional: float        # shares * price
    delta_weight: float    # |target_w - prev_w|
    estimated_cost: float  # transaction cost in dollars
    cost_pct_nav: float    # cost / nav


# ---------------------------------------------------------------------------
# Trade list generation
# ---------------------------------------------------------------------------

def generate_trade_list(
    prev_weights: Dict[str, float],
    target_weights: Dict[str, float],
    prices: Dict[str, float],
    nav: float,
    cost_model: CostModel = DEFAULT_COST_MODEL,
    min_trade_notional: float = 0.0,
) -> pd.DataFrame:
    """
    Generate an executable trade list from weight changes.

    Parameters
    ----------
    prev_weights : dict {ticker: weight}
        Current portfolio weights.
    target_weights : dict {ticker: weight}
        Desired target weights.
    prices : dict {ticker: price}
        Current prices.
    nav : float
        Portfolio NAV.
    cost_model : CostModel
    min_trade_notional : float
        Minimum trade size in dollars (e.g. 100 = $100 min). Smaller trades
        are ignored.

    Returns
    -------
    pd.DataFrame with columns:
        ticker, side, shares, price, notional, delta_weight, estimated_cost, cost_pct_nav
    """
    all_tickers = set(prev_weights) | set(target_weights)
    trades = []

    for ticker in all_tickers:
        prev_w = prev_weights.get(ticker, 0.0)
        tgt_w = target_weights.get(ticker, 0.0)
        price = prices.get(ticker, 0.0)

        if price <= 0:
            continue

        delta_w = tgt_w - prev_w
        if abs(delta_w) < 1e-6:
            continue  # no meaningful trade

        # Compute shares and notional
        delta_dollars = delta_w * nav
        shares = delta_dollars / price

        if abs(delta_dollars) < min_trade_notional:
            continue  # below minimum size

        # Estimate cost
        cost = estimate_one_way_cost(abs(delta_dollars), price, model=cost_model)
        cost_pct = cost / nav if nav > 0 else 0.0

        side = "buy" if delta_w > 0 else "sell"

        trades.append({
            "ticker": ticker,
            "side": side,
            "shares": shares,
            "price": price,
            "notional": delta_dollars,
            "delta_weight": delta_w,
            "estimated_cost": cost,
            "cost_pct_nav": cost_pct,
        })

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def filter_trades_by_cost(
    trade_list: pd.DataFrame,
    min_net_alpha_pct: float = 0.001,
) -> pd.DataFrame:
    """
    Filter out trades where estimated cost exceeds minimum net alpha threshold.

    Skip a trade if:
        cost_pct_nav > min_net_alpha_pct

    The idea: a trade is only worth executing if its transaction cost is small
    relative to the expected alpha benefit (which we approximate as a fraction
    of NAV).

    Parameters
    ----------
    trade_list : pd.DataFrame
        Output from generate_trade_list().
    min_net_alpha_pct : float
        Minimum net alpha (as % of NAV) required to justify a trade.
        E.g. 0.001 = 0.1%. Set to 0 to disable filtering.

    Returns
    -------
    pd.DataFrame — filtered trade list (subset of input).
    """
    if trade_list.empty or min_net_alpha_pct <= 0:
        return trade_list.copy()

    # Keep trades where cost < min_net_alpha_pct of NAV
    filtered = trade_list[trade_list["cost_pct_nav"] <= min_net_alpha_pct].copy()

    return filtered


def summarise_trades(trade_list: pd.DataFrame) -> str:
    """Return a human-readable summary of trades."""
    if trade_list.empty:
        return "No trades to execute."

    lines = [
        "",
        "=" * 70,
        "TRADE LIST SUMMARY",
        "=" * 70,
        f"{'Ticker':<8} {'Side':<5} {'Shares':>12} {'Price':>10} {'Notional':>12} {'Est. Cost':>12}",
        "-" * 70,
    ]

    total_notional = 0.0
    total_cost = 0.0

    for _, row in trade_list.iterrows():
        lines.append(
            f"  {row['ticker']:<6} {row['side']:<5} "
            f"{row['shares']:>12.2f} ${row['price']:>9.2f} "
            f"${row['notional']:>11.2f} ${row['estimated_cost']:>11.2f}"
        )
        total_notional += abs(row["notional"])
        total_cost += row["estimated_cost"]

    lines += [
        "-" * 70,
        f"{'Total':<8} {'':>5} {total_notional:>23.2f} ${total_cost:>11.2f}",
        f"Total one-way turnover: {total_notional:>10.2f}",
        f"Total estimated cost: ${total_cost:>12.2f}",
        "=" * 70,
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trade execution interface (placeholder for broker)
# ---------------------------------------------------------------------------

def submit_orders(
    trade_list: pd.DataFrame,
    broker_client=None,
    dry_run: bool = True,
) -> pd.DataFrame:
    """
    Submit trades to the broker.

    Placeholder: in production, this calls broker API (e.g. Alpaca).
    For now, returns a copy of the trade list with 'order_id' column.

    Parameters
    ----------
    trade_list : pd.DataFrame
        Output from filter_trades_by_cost().
    broker_client : object, optional
        Broker API client (e.g. alpaca_trade_api.REST). If None, dry-run only.
    dry_run : bool
        If True, don't actually submit. Just return the trade list as-is.

    Returns
    -------
    pd.DataFrame — trade list with 'order_id' and 'status' columns.
    """
    result = trade_list.copy()

    if dry_run or broker_client is None:
        result["order_id"] = [f"DRY_RUN_{i:06d}" for i in range(len(result))]
        result["status"] = "dry_run"
        return result

    # In production, iterate over trades and submit to broker
    # For now, placeholder:
    try:
        order_ids = []
        for _, trade in trade_list.iterrows():
            # Example Alpaca API call (requires alpaca_trade_api.REST):
            # order = broker_client.submit_order(
            #     symbol=trade['ticker'],
            #     qty=abs(trade['shares']),
            #     side=trade['side'],
            #     type='market',
            #     time_in_force='day',
            # )
            # order_ids.append(order.id)
            # For now, placeholder:
            order_ids.append(f"NOT_IMPLEMENTED")

        result["order_id"] = order_ids
        result["status"] = "submitted"
    except Exception as e:
        result["order_id"] = None
        result["status"] = f"error: {e}"

    return result


def confirm_fills(
    submitted_trades: pd.DataFrame,
    broker_client=None,
) -> pd.DataFrame:
    """
    Check order status and record actual fill prices.

    Placeholder for broker API calls.

    Parameters
    ----------
    submitted_trades : pd.DataFrame
        Output from submit_orders() with 'order_id' column.
    broker_client : object, optional

    Returns
    -------
    pd.DataFrame — submitted_trades with 'fill_price', 'fill_shares', 'filled_at' columns.
    """
    result = submitted_trades.copy()

    if "fill_price" not in result.columns:
        result["fill_price"] = result.get("price", np.nan)
    if "fill_shares" not in result.columns:
        result["fill_shares"] = result.get("shares", 0.0)
    if "filled_at" not in result.columns:
        result["filled_at"] = None

    if broker_client is None:
        return result

    # In production, query broker for each order_id and record fill details
    # For now, assume all trades filled at the target price:
    # (In reality would check order status, slippage, partial fills, etc.)

    return result
