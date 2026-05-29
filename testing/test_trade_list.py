# test_trade_list.py
# Run from project root: pytest testing/test_trade_list.py -v

import sys
from pathlib import Path

bot_dir = Path(__file__).resolve().parent.parent / "bot"
if str(bot_dir) not in sys.path:
    sys.path.insert(0, str(bot_dir))

import numpy as np
import pandas as pd
import pytest

from execution.trade_list import (
    Trade,
    generate_trade_list,
    filter_trades_by_cost,
    summarise_trades,
    submit_orders,
    confirm_fills,
)
from backtest.costs import CostModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cost_model(spread_bps=5.0):
    """Create a simple cost model with fixed spread."""
    return CostModel(
        spread_bps=spread_bps,
        use_market_impact=False,
    )


# ---------------------------------------------------------------------------
# Trade dataclass
# ---------------------------------------------------------------------------

class TestTradeDataclass:
    def test_trade_construction(self):
        trade = Trade(
            ticker="AAPL",
            side="buy",
            shares=100.0,
            price=150.0,
            notional=15000.0,
            delta_weight=0.10,
            estimated_cost=75.0,
            cost_pct_nav=0.001,
        )
        assert trade.ticker == "AAPL"
        assert trade.side == "buy"
        assert trade.shares == 100.0

    def test_trade_sell_side(self):
        trade = Trade(
            ticker="AAPL",
            side="sell",
            shares=-100.0,
            price=150.0,
            notional=-15000.0,
            delta_weight=-0.10,
            estimated_cost=75.0,
            cost_pct_nav=0.001,
        )
        assert trade.side == "sell"


# ---------------------------------------------------------------------------
# generate_trade_list
# ---------------------------------------------------------------------------

class TestGenerateTradeList:
    def test_simple_buy_trade(self):
        """Generate a single buy trade."""
        prev_weights = {"AAPL": 0.0}
        target_weights = {"AAPL": 0.1}
        prices = {"AAPL": 100.0}
        nav = 100000.0

        trades = generate_trade_list(prev_weights, target_weights, prices, nav)

        assert len(trades) == 1
        assert trades.iloc[0]["ticker"] == "AAPL"
        assert trades.iloc[0]["side"] == "buy"
        assert trades.iloc[0]["shares"] == pytest.approx(100.0)  # (0.1 * 100000) / 100
        assert trades.iloc[0]["notional"] == pytest.approx(10000.0)

    def test_simple_sell_trade(self):
        """Generate a single sell trade."""
        prev_weights = {"AAPL": 0.2}
        target_weights = {"AAPL": 0.05}
        prices = {"AAPL": 100.0}
        nav = 100000.0

        trades = generate_trade_list(prev_weights, target_weights, prices, nav)

        assert len(trades) == 1
        assert trades.iloc[0]["side"] == "sell"
        assert trades.iloc[0]["notional"] == pytest.approx(-15000.0)  # (0.05 - 0.2) * 100000

    def test_no_trades_when_weights_unchanged(self):
        """Empty trade list when weights don't change."""
        prev_weights = {"AAPL": 0.1}
        target_weights = {"AAPL": 0.1}
        prices = {"AAPL": 100.0}
        nav = 100000.0

        trades = generate_trade_list(prev_weights, target_weights, prices, nav)

        assert trades.empty

    def test_multiple_tickers(self):
        """Generate trades for multiple tickers."""
        prev_weights = {"AAPL": 0.1, "MSFT": 0.1, "GOOG": 0.0}
        target_weights = {"AAPL": 0.15, "MSFT": 0.05, "GOOG": 0.2}
        prices = {"AAPL": 100.0, "MSFT": 200.0, "GOOG": 50.0}
        nav = 100000.0

        trades = generate_trade_list(prev_weights, target_weights, prices, nav)

        assert len(trades) == 3
        assert set(trades["ticker"]) == {"AAPL", "MSFT", "GOOG"}

        aapl = trades[trades["ticker"] == "AAPL"].iloc[0]
        assert aapl["side"] == "buy"

        msft = trades[trades["ticker"] == "MSFT"].iloc[0]
        assert msft["side"] == "sell"

        goog = trades[trades["ticker"] == "GOOG"].iloc[0]
        assert goog["side"] == "buy"

    def test_cost_estimation(self):
        """Verify cost is estimated using the cost model."""
        prev_weights = {"AAPL": 0.0}
        target_weights = {"AAPL": 0.1}
        prices = {"AAPL": 100.0}
        nav = 100000.0
        cost_model = _make_cost_model(spread_bps=10.0)  # 10bps = 0.1%

        trades = generate_trade_list(prev_weights, target_weights, prices, nav, cost_model=cost_model)

        # For a $10,000 trade at 100bps spread, cost ≈ $50 (half spread)
        assert trades.iloc[0]["estimated_cost"] > 0
        assert trades.iloc[0]["cost_pct_nav"] > 0

    def test_cost_pct_nav(self):
        """Verify cost_pct_nav is calculated correctly."""
        prev_weights = {"AAPL": 0.0}
        target_weights = {"AAPL": 0.1}
        prices = {"AAPL": 100.0}
        nav = 100000.0
        cost_model = _make_cost_model(spread_bps=20.0)  # 20bps

        trades = generate_trade_list(prev_weights, target_weights, prices, nav, cost_model=cost_model)

        # cost_pct_nav = cost / nav
        row = trades.iloc[0]
        assert abs(row["cost_pct_nav"] - row["estimated_cost"] / nav) < 1e-6

    def test_min_trade_notional_filtering(self):
        """Skip trades below minimum notional."""
        prev_weights = {"AAPL": 0.0, "MSFT": 0.0}
        target_weights = {"AAPL": 0.01, "MSFT": 0.1}  # AAPL $1k, MSFT $10k
        prices = {"AAPL": 100.0, "MSFT": 100.0}
        nav = 100000.0

        trades = generate_trade_list(prev_weights, target_weights, prices, nav, min_trade_notional=5000.0)

        # AAPL $1k trade filtered out, MSFT $10k included
        assert len(trades) == 1
        assert trades.iloc[0]["ticker"] == "MSFT"

    def test_zero_or_negative_price_skipped(self):
        """Skip trades for tickers with zero or negative price."""
        prev_weights = {"AAPL": 0.0, "BAD": 0.0}
        target_weights = {"AAPL": 0.1, "BAD": 0.1}
        prices = {"AAPL": 100.0, "BAD": 0.0}
        nav = 100000.0

        trades = generate_trade_list(prev_weights, target_weights, prices, nav)

        assert len(trades) == 1
        assert trades.iloc[0]["ticker"] == "AAPL"

    def test_delta_weight_recorded(self):
        """Verify delta_weight is recorded correctly."""
        prev_weights = {"AAPL": 0.1}
        target_weights = {"AAPL": 0.25}
        prices = {"AAPL": 100.0}
        nav = 100000.0

        trades = generate_trade_list(prev_weights, target_weights, prices, nav)

        assert trades.iloc[0]["delta_weight"] == pytest.approx(0.15)

    def test_new_position_from_zero(self):
        """Add a position from zero."""
        prev_weights = {}
        target_weights = {"AAPL": 0.1}
        prices = {"AAPL": 100.0}
        nav = 100000.0

        trades = generate_trade_list(prev_weights, target_weights, prices, nav)

        assert len(trades) == 1
        assert trades.iloc[0]["side"] == "buy"

    def test_liquidate_to_zero(self):
        """Liquidate a position to zero."""
        prev_weights = {"AAPL": 0.2}
        target_weights = {"AAPL": 0.0}
        prices = {"AAPL": 100.0}
        nav = 100000.0

        trades = generate_trade_list(prev_weights, target_weights, prices, nav)

        assert len(trades) == 1
        assert trades.iloc[0]["side"] == "sell"


# ---------------------------------------------------------------------------
# filter_trades_by_cost
# ---------------------------------------------------------------------------

class TestFilterTradesByCost:
    def test_no_filtering_when_threshold_zero(self):
        """No trades filtered when threshold is 0."""
        trade_list = pd.DataFrame({
            "ticker": ["AAPL"],
            "side": ["buy"],
            "shares": [100.0],
            "price": [100.0],
            "notional": [10000.0],
            "delta_weight": [0.1],
            "estimated_cost": [100.0],
            "cost_pct_nav": [0.001],
        })

        filtered = filter_trades_by_cost(trade_list, min_net_alpha_pct=0.0)

        assert len(filtered) == 1

    def test_empty_trade_list(self):
        """Filtering empty list returns empty."""
        trade_list = pd.DataFrame()

        filtered = filter_trades_by_cost(trade_list)

        assert filtered.empty

    def test_filter_expensive_trade(self):
        """Filter out a trade where cost > threshold."""
        trade_list = pd.DataFrame({
            "ticker": ["AAPL"],
            "side": ["buy"],
            "shares": [1000.0],
            "price": [100.0],
            "notional": [100000.0],
            "delta_weight": [0.5],
            "estimated_cost": [1000.0],
            "cost_pct_nav": [0.01],  # 1% cost
        })

        # Threshold 0.5% - this 1% cost should be filtered
        filtered = filter_trades_by_cost(trade_list, min_net_alpha_pct=0.005)

        assert filtered.empty

    def test_keep_cheap_trade(self):
        """Keep a trade where cost <= threshold."""
        trade_list = pd.DataFrame({
            "ticker": ["AAPL"],
            "side": ["buy"],
            "shares": [100.0],
            "price": [100.0],
            "notional": [10000.0],
            "delta_weight": [0.05],
            "estimated_cost": [25.0],
            "cost_pct_nav": [0.00025],  # 0.025% cost
        })

        # Threshold 0.1% - this 0.025% cost passes
        filtered = filter_trades_by_cost(trade_list, min_net_alpha_pct=0.001)

        assert len(filtered) == 1

    def test_filter_mixed_trades(self):
        """Filter expensive trades from mixed list."""
        trade_list = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "side": ["buy", "buy", "sell"],
            "shares": [100.0, 200.0, 50.0],
            "price": [100.0, 100.0, 100.0],
            "notional": [10000.0, 20000.0, 5000.0],
            "delta_weight": [0.05, 0.1, 0.025],
            "estimated_cost": [50.0, 500.0, 25.0],
            "cost_pct_nav": [0.0005, 0.005, 0.00025],
        })

        # Threshold 0.1% (0.001) - MSFT at 0.5% should be filtered
        filtered = filter_trades_by_cost(trade_list, min_net_alpha_pct=0.001)

        assert len(filtered) == 2
        assert "MSFT" not in filtered["ticker"].values
        assert set(filtered["ticker"]) == {"AAPL", "GOOG"}

    def test_boundary_at_threshold(self):
        """Trade at exactly the threshold should be kept."""
        trade_list = pd.DataFrame({
            "ticker": ["AAPL"],
            "side": ["buy"],
            "shares": [100.0],
            "price": [100.0],
            "notional": [10000.0],
            "delta_weight": [0.05],
            "estimated_cost": [50.0],
            "cost_pct_nav": [0.0005],  # Exactly 0.1%
        })

        filtered = filter_trades_by_cost(trade_list, min_net_alpha_pct=0.0005)

        assert len(filtered) == 1


# ---------------------------------------------------------------------------
# summarise_trades
# ---------------------------------------------------------------------------

class TestSummariseTrades:
    def test_summary_nonempty(self):
        """Generate summary for non-empty trade list."""
        trade_list = pd.DataFrame({
            "ticker": ["AAPL"],
            "side": ["buy"],
            "shares": [100.0],
            "price": [150.0],
            "notional": [15000.0],
            "delta_weight": [0.1],
            "estimated_cost": [75.0],
            "cost_pct_nav": [0.001],
        })

        summary = summarise_trades(trade_list)

        assert isinstance(summary, str)
        assert "AAPL" in summary
        assert "buy" in summary
        assert "100.00" in summary

    def test_summary_empty(self):
        """Summary for empty trade list."""
        trade_list = pd.DataFrame()

        summary = summarise_trades(trade_list)

        assert "No trades" in summary

    def test_summary_contains_totals(self):
        """Summary includes total notional and cost."""
        trade_list = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "side": ["buy", "sell"],
            "shares": [100.0, 50.0],
            "price": [150.0, 200.0],
            "notional": [15000.0, 10000.0],
            "delta_weight": [0.1, 0.05],
            "estimated_cost": [75.0, 50.0],
            "cost_pct_nav": [0.001, 0.0005],
        })

        summary = summarise_trades(trade_list)

        # Totals: notional = 25000, cost = 125
        assert "25000.00" in summary
        assert "125.00" in summary


# ---------------------------------------------------------------------------
# submit_orders
# ---------------------------------------------------------------------------

class TestSubmitOrders:
    def test_dry_run_mode(self):
        """Dry run returns trade list with order IDs."""
        trade_list = pd.DataFrame({
            "ticker": ["AAPL"],
            "side": ["buy"],
            "shares": [100.0],
            "price": [150.0],
            "notional": [15000.0],
            "delta_weight": [0.1],
            "estimated_cost": [75.0],
            "cost_pct_nav": [0.001],
        })

        result = submit_orders(trade_list, dry_run=True)

        assert "order_id" in result.columns
        assert "status" in result.columns
        assert result.iloc[0]["status"] == "dry_run"
        assert "DRY_RUN" in result.iloc[0]["order_id"]

    def test_dry_run_no_broker_client(self):
        """Dry run when no broker client provided."""
        trade_list = pd.DataFrame({
            "ticker": ["AAPL"],
            "side": ["buy"],
            "shares": [100.0],
            "price": [150.0],
            "notional": [15000.0],
            "delta_weight": [0.1],
            "estimated_cost": [75.0],
            "cost_pct_nav": [0.001],
        })

        result = submit_orders(trade_list, broker_client=None)

        assert len(result) == 1
        assert result.iloc[0]["status"] == "dry_run"

    def test_multiple_orders_dry_run(self):
        """Multiple trades in dry run get unique order IDs."""
        trade_list = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "side": ["buy", "sell", "buy"],
            "shares": [100.0, 50.0, 75.0],
            "price": [150.0, 200.0, 100.0],
            "notional": [15000.0, 10000.0, 7500.0],
            "delta_weight": [0.1, 0.05, 0.05],
            "estimated_cost": [75.0, 50.0, 37.5],
            "cost_pct_nav": [0.001, 0.0005, 0.0004],
        })

        result = submit_orders(trade_list, dry_run=True)

        assert len(result) == 3
        order_ids = result["order_id"].tolist()
        assert len(set(order_ids)) == 3  # All unique

    def test_dry_run_preserves_trade_columns(self):
        """Dry run preserves original trade columns."""
        trade_list = pd.DataFrame({
            "ticker": ["AAPL"],
            "side": ["buy"],
            "shares": [100.0],
            "price": [150.0],
            "notional": [15000.0],
            "delta_weight": [0.1],
            "estimated_cost": [75.0],
            "cost_pct_nav": [0.001],
        })

        result = submit_orders(trade_list, dry_run=True)

        assert "ticker" in result.columns
        assert "side" in result.columns
        assert result.iloc[0]["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# confirm_fills
# ---------------------------------------------------------------------------

class TestConfirmFills:
    def test_fill_columns_created(self):
        """Confirm fills initializes fill columns if missing."""
        submitted = pd.DataFrame({
            "ticker": ["AAPL"],
            "side": ["buy"],
            "shares": [100.0],
            "price": [150.0],
            "notional": [15000.0],
            "delta_weight": [0.1],
            "estimated_cost": [75.0],
            "cost_pct_nav": [0.001],
            "order_id": ["DRY_RUN_000001"],
            "status": ["dry_run"],
        })

        result = confirm_fills(submitted)

        assert "fill_price" in result.columns
        assert "fill_shares" in result.columns
        assert "filled_at" in result.columns

    def test_fill_price_defaults_to_price(self):
        """Fill price defaults to order price if not present."""
        submitted = pd.DataFrame({
            "ticker": ["AAPL"],
            "side": ["buy"],
            "shares": [100.0],
            "price": [150.0],
            "notional": [15000.0],
            "delta_weight": [0.1],
            "estimated_cost": [75.0],
            "cost_pct_nav": [0.001],
            "order_id": ["DRY_RUN_000001"],
            "status": ["dry_run"],
        })

        result = confirm_fills(submitted)

        assert result.iloc[0]["fill_price"] == 150.0

    def test_fill_shares_defaults_to_shares(self):
        """Fill shares defaults to ordered shares if not present."""
        submitted = pd.DataFrame({
            "ticker": ["AAPL"],
            "side": ["buy"],
            "shares": [100.0],
            "price": [150.0],
            "notional": [15000.0],
            "delta_weight": [0.1],
            "estimated_cost": [75.0],
            "cost_pct_nav": [0.001],
            "order_id": ["DRY_RUN_000001"],
            "status": ["dry_run"],
        })

        result = confirm_fills(submitted)

        assert result.iloc[0]["fill_shares"] == 100.0

    def test_filled_at_initialized_none(self):
        """Filled_at initialized to None."""
        submitted = pd.DataFrame({
            "ticker": ["AAPL"],
            "side": ["buy"],
            "shares": [100.0],
            "price": [150.0],
            "notional": [15000.0],
            "delta_weight": [0.1],
            "estimated_cost": [75.0],
            "cost_pct_nav": [0.001],
            "order_id": ["DRY_RUN_000001"],
            "status": ["dry_run"],
        })

        result = confirm_fills(submitted)

        assert result.iloc[0]["filled_at"] is None

    def test_existing_fill_columns_preserved(self):
        """If fill columns exist, they are preserved."""
        submitted = pd.DataFrame({
            "ticker": ["AAPL"],
            "side": ["buy"],
            "shares": [100.0],
            "price": [150.0],
            "notional": [15000.0],
            "delta_weight": [0.1],
            "estimated_cost": [75.0],
            "cost_pct_nav": [0.001],
            "order_id": ["DRY_RUN_000001"],
            "status": ["dry_run"],
            "fill_price": [149.5],
            "fill_shares": [100.0],
            "filled_at": ["2026-01-15"],
        })

        result = confirm_fills(submitted)

        assert result.iloc[0]["fill_price"] == 149.5
        assert result.iloc[0]["filled_at"] == "2026-01-15"

    def test_no_broker_client(self):
        """Confirm fills with no broker client returns filled data."""
        submitted = pd.DataFrame({
            "ticker": ["AAPL"],
            "side": ["buy"],
            "shares": [100.0],
            "price": [150.0],
            "notional": [15000.0],
            "delta_weight": [0.1],
            "estimated_cost": [75.0],
            "cost_pct_nav": [0.001],
            "order_id": ["DRY_RUN_000001"],
            "status": ["dry_run"],
        })

        result = confirm_fills(submitted, broker_client=None)

        assert "fill_price" in result.columns
        assert result.iloc[0]["fill_price"] == 150.0

    def test_multiple_fills(self):
        """Confirm fills for multiple orders."""
        submitted = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "side": ["buy", "sell"],
            "shares": [100.0, 50.0],
            "price": [150.0, 200.0],
            "notional": [15000.0, 10000.0],
            "delta_weight": [0.1, 0.05],
            "estimated_cost": [75.0, 50.0],
            "cost_pct_nav": [0.001, 0.0005],
            "order_id": ["DRY_RUN_000001", "DRY_RUN_000002"],
            "status": ["dry_run", "dry_run"],
        })

        result = confirm_fills(submitted)

        assert len(result) == 2
        assert result.iloc[0]["fill_price"] == 150.0
        assert result.iloc[1]["fill_price"] == 200.0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline_buy_only(self):
        """Full pipeline: generate → filter → summarize."""
        prev_weights = {"AAPL": 0.0, "MSFT": 0.0}
        target_weights = {"AAPL": 0.1, "MSFT": 0.05}
        prices = {"AAPL": 100.0, "MSFT": 200.0}
        nav = 100000.0

        trades = generate_trade_list(prev_weights, target_weights, prices, nav)
        filtered = filter_trades_by_cost(trades, min_net_alpha_pct=0.01)
        summary = summarise_trades(filtered)

        assert len(filtered) == 2
        assert "AAPL" in summary
        assert "MSFT" in summary

    def test_full_pipeline_with_filtering(self):
        """Full pipeline with cost-based filtering."""
        prev_weights = {"AAPL": 0.0, "MSFT": 0.0}
        target_weights = {"AAPL": 0.5, "MSFT": 0.05}  # AAPL expensive
        prices = {"AAPL": 100.0, "MSFT": 200.0}
        nav = 100000.0
        cost_model = _make_cost_model(spread_bps=50.0)  # High cost

        trades = generate_trade_list(prev_weights, target_weights, prices, nav, cost_model=cost_model)
        filtered = filter_trades_by_cost(trades, min_net_alpha_pct=0.001)  # Tight threshold

        # AAPL large trade might be filtered, MSFT small trade should pass
        assert len(filtered) >= 1

    def test_execution_pipeline_dry_run(self):
        """Full execution pipeline in dry-run mode."""
        prev_weights = {"AAPL": 0.0}
        target_weights = {"AAPL": 0.1}
        prices = {"AAPL": 100.0}
        nav = 100000.0

        trades = generate_trade_list(prev_weights, target_weights, prices, nav)
        filtered = filter_trades_by_cost(trades, min_net_alpha_pct=0.001)
        submitted = submit_orders(filtered, dry_run=True)
        confirmed = confirm_fills(submitted)

        assert len(confirmed) == 1
        assert "order_id" in confirmed.columns
        assert "fill_price" in confirmed.columns
        assert confirmed.iloc[0]["status"] == "dry_run"
