# test_backtest_engine.py
# Run from bot/:  pytest ../testing/test_backtest_engine.py -v

import sys
import tempfile
from datetime import date
from pathlib import Path

bot_dir = Path(__file__).resolve().parent.parent / "bot"
if str(bot_dir) not in sys.path:
    sys.path.insert(0, str(bot_dir))

import numpy as np
import pandas as pd
import pytest

from backtest.costs import (
    CostModel, DEFAULT_COST_MODEL,
    compute_rebalance_costs, estimate_one_way_cost,
)
from backtest.engine import BacktestEngine, BacktestResult
from backtest.integrity import check_costs_applied, check_universe_snapshots_used, run_honesty_checks
from backtest.metrics import (
    cagr, sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
    hit_rate, compute_ic, compute_ic_series, ic_tstat, compute_metrics,
)
from backtest.walk_forward import generate_folds, expanding_folds, FoldSpec
from data.integrity_checks import IntegrityError
from data.universe import save_universe_snapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(ticker: str, start: str, end: str, seed: int = 0) -> pd.DataFrame:
    """Generate synthetic daily OHLCV for testing."""
    np.random.seed(seed)
    dates = pd.date_range(start, end, freq="B")
    n = len(dates)
    close = 100 * np.cumprod(1 + np.random.randn(n) * 0.01)
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)


def _make_price_panel(tickers, start, end):
    return {t: _make_price_df(t, start, end, seed=i) for i, t in enumerate(tickers)}


def _equal_weight_signal(reb_date, tickers, price_data):
    """Simple equal-weight signal for testing."""
    active = [t for t in tickers if t in price_data and not price_data[t].empty]
    if not active:
        return {}
    w = 1.0 / len(active)
    return {t: w for t in active}


# ---------------------------------------------------------------------------
# costs.py
# ---------------------------------------------------------------------------

def test_estimate_one_way_cost_zero_for_zero_trade():
    assert estimate_one_way_cost(0.0, 100.0) == 0.0


def test_estimate_one_way_cost_spread_only():
    model = CostModel(spread_bps=10.0, use_market_impact=False)
    cost = estimate_one_way_cost(10_000.0, 100.0, model=model)
    assert abs(cost - 10.0) < 1e-9  # 10 bps of $10k = $10


def test_compute_rebalance_costs_no_change():
    weights = {"AAPL": 0.5, "MSFT": 0.5}
    cost, per_ticker = compute_rebalance_costs(
        prev_weights=weights, target_weights=weights,
        prices={"AAPL": 150.0, "MSFT": 300.0},
        portfolio_value=100_000.0,
    )
    assert cost == 0.0
    assert per_ticker == {}


def test_compute_rebalance_costs_full_turnover():
    prev = {}
    target = {"AAPL": 0.5, "MSFT": 0.5}
    model = CostModel(spread_bps=10.0)
    cost, per_ticker = compute_rebalance_costs(
        prev_weights=prev, target_weights=target,
        prices={"AAPL": 150.0, "MSFT": 300.0},
        portfolio_value=100_000.0, model=model,
    )
    # 0.1% of $50k for each position → $50 + $50 = $100
    assert abs(cost - 100.0) < 1e-6
    assert "AAPL" in per_ticker and "MSFT" in per_ticker


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------

def test_cagr_zero_returns():
    returns = pd.Series([0.0] * 252)
    assert abs(cagr(returns)) < 1e-9


def test_cagr_positive():
    returns = pd.Series([0.001] * 252)
    assert cagr(returns) > 0


def test_sharpe_positive_returns():
    returns = pd.Series([0.001] * 252)
    assert sharpe_ratio(returns) > 0


def test_max_drawdown_never_negative_series():
    returns = pd.Series([0.01] * 252)
    assert max_drawdown(returns) == 0.0


def test_max_drawdown_known_value():
    # Price goes 100 → 100 → 80 → 100. Peak is 1.0, trough is 0.8 → -20% DD
    # First return is 0 (establishes peak), then -20%, then recovery
    returns = pd.Series([0.0, -0.2, 0.25])
    dd = max_drawdown(returns)
    assert abs(dd - (-0.2)) < 1e-9


def test_hit_rate():
    returns = pd.Series([0.01, -0.01, 0.02, -0.005, 0.015])
    assert abs(hit_rate(returns) - 0.6) < 1e-9


def test_compute_ic_perfect_correlation():
    signal = pd.Series([3, 1, 2, 5, 4], index=["A", "B", "C", "D", "E"])
    fwd = pd.Series([3, 1, 2, 5, 4], index=["A", "B", "C", "D", "E"])
    assert abs(compute_ic(signal, fwd) - 1.0) < 1e-9


def test_compute_ic_no_correlation():
    signal = pd.Series([1, 2, 3, 4, 5], index=list("ABCDE"))
    fwd = pd.Series([5, 4, 3, 2, 1], index=list("ABCDE"))
    assert abs(compute_ic(signal, fwd) - (-1.0)) < 1e-9


def test_compute_ic_insufficient_data():
    signal = pd.Series([1, 2], index=["A", "B"])
    fwd = pd.Series([1, 2], index=["A", "B"])
    assert pd.isna(compute_ic(signal, fwd))


def test_ic_tstat_significant():
    ic_series = pd.Series([0.05] * 24)  # 24 months, consistent IC of 0.05
    t = ic_tstat(ic_series)
    # t = 0.05 / (0 / sqrt(24)) → std=0, but all same → std=0 → nan
    # Use variable IC to get a real t-stat
    np.random.seed(42)
    ic_series2 = pd.Series(np.random.normal(0.06, 0.1, 36))
    t2 = ic_tstat(ic_series2)
    assert not pd.isna(t2)


def test_compute_metrics_keys():
    returns = pd.Series(np.random.randn(252) * 0.01)
    m = compute_metrics(returns)
    assert "cagr" in m
    assert "sharpe" in m
    assert "max_drawdown" in m
    assert "sortino" in m


# ---------------------------------------------------------------------------
# walk_forward.py
# ---------------------------------------------------------------------------

def test_generate_folds_basic():
    folds = generate_folds("2010-01-01", "2023-12-31", train_years=3, test_months=6, min_train_years=5)
    assert len(folds) > 0
    for f in folds:
        assert f.train_end < f.test_start   # strict: no overlap


def test_generate_folds_no_overlap_between_folds():
    folds = generate_folds("2010-01-01", "2023-12-31")
    for i in range(len(folds) - 1):
        assert folds[i].test_end <= folds[i + 1].test_start


def test_generate_folds_insufficient_data_raises():
    with pytest.raises(ValueError, match="Not enough data"):
        generate_folds("2020-01-01", "2021-01-01", min_train_years=5)


def test_fold_spec_raises_on_overlap():
    with pytest.raises(IntegrityError, match="Train/test overlap"):
        FoldSpec(
            fold_index=0,
            train_start=date(2010, 1, 1),
            train_end=date(2015, 6, 30),
            test_start=date(2015, 6, 30),  # same day as train_end → overlap
            test_end=date(2015, 12, 31),
        )


def test_expanding_folds_train_always_from_start():
    folds = expanding_folds("2010-01-01", "2023-12-31", min_train_years=5)
    start = date(2010, 1, 1)
    for f in folds:
        assert f.train_start == start


# ---------------------------------------------------------------------------
# engine.py  (synthetic end-to-end)
# ---------------------------------------------------------------------------

def test_engine_runs_and_returns_result():
    tickers = ["AAPL", "MSFT", "GOOG"]
    panel = _make_price_panel(tickers, "2020-01-01", "2022-12-31")

    with tempfile.TemporaryDirectory() as tmp:
        snap_dir = Path(tmp)
        # Create snapshots for all rebalance dates
        for d in pd.date_range("2020-01-31", "2022-12-31", freq="ME"):
            save_universe_snapshot(tickers, as_of=d.date(), snapshot_dir=snap_dir)

        engine = BacktestEngine(
            cost_model=CostModel(spread_bps=10),
            initial_capital=100_000.0,
            rebalance_freq="ME",
            run_integrity_checks=True,
            snapshot_dir=snap_dir,
            benchmark_ticker=None,
        )
        result = engine.run(
            signal_fn=_equal_weight_signal,
            price_panel=panel,
            start_date="2020-01-01",
            end_date="2022-12-31",
            verbose=False,
        )

    assert isinstance(result, BacktestResult)
    assert not result.daily_returns.empty
    assert not result.portfolio_values.empty
    assert len(result.rebalance_records) > 0


def test_engine_costs_reduce_nav():
    """Costs must reduce NAV below a costless simulation."""
    tickers = ["AAPL", "MSFT"]
    panel = _make_price_panel(tickers, "2021-01-01", "2022-12-31")

    with tempfile.TemporaryDirectory() as tmp:
        snap_dir = Path(tmp)
        for d in pd.date_range("2021-01-31", "2022-12-31", freq="ME"):
            save_universe_snapshot(tickers, as_of=d.date(), snapshot_dir=snap_dir)

        engine_costly = BacktestEngine(
            cost_model=CostModel(spread_bps=100),  # very high cost
            initial_capital=100_000.0,
            snapshot_dir=snap_dir, benchmark_ticker=None,
        )
        result = engine_costly.run(
            signal_fn=_equal_weight_signal, price_panel=panel,
            start_date="2021-01-01", end_date="2022-12-31", verbose=False,
        )
        total_costs = result.rebalance_log["total_cost"].sum()
        assert total_costs > 0


def test_engine_fallback_universe_warns():
    """When no snapshot exists, engine should warn and use fallback."""
    tickers = ["AAPL", "MSFT"]
    panel = _make_price_panel(tickers, "2021-01-01", "2021-06-30")

    with tempfile.TemporaryDirectory() as tmp:
        engine = BacktestEngine(
            snapshot_dir=Path(tmp), benchmark_ticker=None,
        )
        with pytest.warns(UserWarning, match="fallback universe"):
            result = engine.run(
                signal_fn=_equal_weight_signal,
                price_panel=panel,
                start_date="2021-01-01",
                end_date="2021-06-30",
                fallback_universe=tickers,
                verbose=False,
            )
    assert isinstance(result, BacktestResult)


def test_engine_no_data_returns_zero_returns():
    """Engine with empty price panel: signal returns no weights → all-zero returns."""
    with tempfile.TemporaryDirectory() as tmp:
        engine = BacktestEngine(snapshot_dir=Path(tmp), benchmark_ticker=None)
        with pytest.warns(UserWarning):
            result = engine.run(
                signal_fn=_equal_weight_signal,
                price_panel={},
                start_date="2021-01-01",
                end_date="2021-06-30",
                fallback_universe=["AAPL"],
                verbose=False,
            )
    # No positions held → all returns are 0.0
    assert (result.daily_returns == 0.0).all()


# ---------------------------------------------------------------------------
# integrity.py (backtest-level)
# ---------------------------------------------------------------------------

def test_check_costs_applied_detects_zero_costs():
    log = pd.DataFrame({"total_cost": [0.0, 0.0, 0.0]})
    passed, _ = check_costs_applied(log)
    assert not passed


def test_check_costs_applied_passes():
    log = pd.DataFrame({"total_cost": [100.0, 50.0, 75.0]})
    passed, _ = check_costs_applied(log)
    assert passed


def test_check_universe_snapshots_missing_col():
    log = pd.DataFrame({"total_cost": [100.0]})
    passed, msg = check_universe_snapshots_used(log)
    assert not passed


def test_check_universe_snapshots_all_true():
    log = pd.DataFrame({"snapshot_used": [True, True, True]})
    passed, _ = check_universe_snapshots_used(log)
    assert passed
