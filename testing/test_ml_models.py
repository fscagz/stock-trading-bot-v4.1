# test_ml_models.py
# Run from project root: pytest testing/test_ml_models.py -v

import sys
from datetime import date
from pathlib import Path

bot_dir = Path(__file__).resolve().parent.parent / "bot"
if str(bot_dir) not in sys.path:
    sys.path.insert(0, str(bot_dir))

import numpy as np
import pandas as pd
import pytest

from models.targets import (
    compute_forward_return,
    compute_forward_returns,
    compute_forward_quantile_labels,
    forward_returns_as_dataframe,
    labels_as_dataframe,
)
from models.walk_forward_ml import WalkForwardRunner, WalkForwardResult
from models.sklearn_models import LightGBMRanker, RidgeFactorModel, get_model
from backtest.walk_forward import FoldSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_series(n=100, seed=0, start="2020-01-01") -> pd.Series:
    np.random.seed(seed)
    dates = pd.date_range(start, periods=n, freq="B")
    prices = 100 * np.cumprod(1 + np.random.randn(n) * 0.01)
    return pd.Series(prices, index=dates, name="close")


def _make_panel(tickers=None, n=100) -> dict:
    tickers = tickers or ["A", "B", "C", "D", "E"]
    return {
        t: pd.DataFrame(
            {"close": _make_price_series(n, seed=i)},
        )
        for i, t in enumerate(tickers)
    }


def _make_features(n=10, n_features=5, seed=0) -> dict:
    np.random.seed(seed)
    tickers = [f"T{i}" for i in range(n)]
    dates = pd.date_range("2020-01-01", periods=10, freq="ME")
    result = {}
    for dt in dates:
        X = pd.DataFrame(
            np.random.randn(n, n_features),
            index=tickers,
            columns=[f"F{j}" for j in range(n_features)],
        )
        result[dt.date()] = X
    return result


def _make_labels(n=10, seed=0) -> dict:
    np.random.seed(seed)
    tickers = [f"T{i}" for i in range(n)]
    dates = pd.date_range("2020-01-01", periods=10, freq="ME")
    result = {}
    for dt in dates:
        y = pd.Series(
            np.random.choice([1, 2, 3, 4, 5], n),
            index=tickers,
        )
        result[dt.date()] = y
    return result


# ---------------------------------------------------------------------------
# targets.py
# ---------------------------------------------------------------------------

class TestComputeForwardReturn:
    def test_positive_return(self):
        prices = pd.Series([100.0, 105.0, 110.0], index=pd.date_range("2020-01-01", periods=3))
        r = compute_forward_return(prices, prices.index[0], 2)
        assert abs(r - 0.10) < 1e-6

    def test_none_when_insufficient_data(self):
        prices = pd.Series([100.0], index=pd.date_range("2020-01-01", periods=1))
        r = compute_forward_return(prices, prices.index[0], 100)
        assert r is None

    def test_zero_price_returns_none(self):
        prices = pd.Series([0.0, 100.0], index=pd.date_range("2020-01-01", periods=2))
        r = compute_forward_return(prices, prices.index[0], 1)
        assert r is None


class TestComputeForwardReturns:
    def test_returns_dict(self):
        panel = _make_panel(tickers=["A", "B", "C"], n=200)
        reb_dates = pd.date_range("2020-01-01", "2020-06-01", freq="ME")
        result = compute_forward_returns(panel, reb_dates.tolist())
        assert isinstance(result, dict)
        assert "A" in result or len(result) == 0

    def test_structure(self):
        panel = _make_panel(n=200)
        reb_dates = [pd.Timestamp("2020-02-28")]
        result = compute_forward_returns(panel, reb_dates)
        # Each ticker should have a dict of {date: return}
        for ticker, ticker_dict in result.items():
            assert isinstance(ticker_dict, dict)


class TestComputeQuantileLabels:
    def test_returns_dict(self):
        fwd = {
            "A": {date(2020, 1, 1): 0.05},
            "B": {date(2020, 1, 1): 0.02},
            "C": {date(2020, 1, 1): -0.01},
        }
        labels = compute_forward_quantile_labels(fwd, n_quantiles=3)
        assert isinstance(labels, dict)

    def test_labels_in_range(self):
        fwd = {
            "A": {date(2020, 1, 1): 0.05},
            "B": {date(2020, 1, 1): 0.02},
            "C": {date(2020, 1, 1): -0.01},
        }
        labels = compute_forward_quantile_labels(fwd, n_quantiles=5)
        for ticker, ticker_dict in labels.items():
            for label in ticker_dict.values():
                assert 1 <= label <= 5

    def test_best_return_gets_top_label(self):
        fwd = {
            "A": {date(2020, 1, 1): 0.10},  # Best
            "B": {date(2020, 1, 1): 0.05},
            "C": {date(2020, 1, 1): -0.05},  # Worst
        }
        labels = compute_forward_quantile_labels(fwd, n_quantiles=3)
        assert labels["A"][date(2020, 1, 1)] > labels["C"][date(2020, 1, 1)]


class TestDataFrameConversion:
    def test_as_dataframe_shape(self):
        fwd = {
            "A": {date(2020, 1, 1): 0.05, date(2020, 2, 1): 0.03},
            "B": {date(2020, 1, 1): 0.02, date(2020, 2, 1): 0.01},
        }
        df = forward_returns_as_dataframe(fwd)
        assert df.shape == (2, 2)

    def test_labels_dataframe(self):
        labels = {
            "A": {date(2020, 1, 1): 5, date(2020, 2, 1): 4},
            "B": {date(2020, 1, 1): 1, date(2020, 2, 1): 2},
        }
        df = labels_as_dataframe(labels)
        assert df.shape == (2, 2)
        assert df.iloc[0, 0] in [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# sklearn_models.py
# ---------------------------------------------------------------------------

class TestRidgeFactorModel:
    def test_fit_and_predict(self):
        X = pd.DataFrame(np.random.randn(50, 5), columns=[f"F{i}" for i in range(5)])
        y = pd.Series(np.random.randn(50))
        model = RidgeFactorModel(alpha=1.0)
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert len(preds) == 10

    def test_feature_importance(self):
        X = pd.DataFrame(np.random.randn(50, 5), columns=[f"F{i}" for i in range(5)])
        y = pd.Series(np.random.randn(50))
        model = RidgeFactorModel()
        model.fit(X, y)
        imp = model.feature_importance_
        assert len(imp) == 5
        assert all(imp >= 0)

    def test_series_y(self):
        # Test that Series y works (most common case)
        X = pd.DataFrame(np.random.randn(50, 3), columns=[f"F{i}" for i in range(3)])
        y = pd.Series(np.random.randn(50))
        model = RidgeFactorModel()
        model.fit(X, y)
        assert model._model is not None
        assert len(model.feature_importance_) == 3


class TestLightGBMRanker:
    def test_requires_lightgbm(self):
        try:
            import lightgbm
            # LightGBM is available
            model = LightGBMRanker(n_leaves=8)
            assert model.n_leaves == 8
        except ImportError:
            pytest.skip("lightgbm not installed")

    def test_fit_and_predict(self):
        try:
            import lightgbm
        except ImportError:
            pytest.skip("lightgbm not installed")

        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"F{i}" for i in range(5)])
        y = pd.Series(np.random.choice([1, 2, 3, 4, 5], 100))
        model = LightGBMRanker(n_estimators=10, verbose=-1)
        model.fit(X, y)
        preds = model.predict(X[:20])
        assert len(preds) == 20

    def test_feature_importance(self):
        try:
            import lightgbm
        except ImportError:
            pytest.skip("lightgbm not installed")

        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"F{i}" for i in range(5)])
        y = pd.Series(np.random.choice([1, 2, 3, 4, 5], 100))
        model = LightGBMRanker(n_estimators=10, verbose=-1)
        model.fit(X, y)
        imp = model.feature_importance_
        assert len(imp) == 5


class TestGetModel:
    def test_ridge(self):
        model = get_model("ridge", alpha=0.5)
        assert isinstance(model, RidgeFactorModel)
        assert model.alpha == 0.5

    def test_lgb(self):
        try:
            import lightgbm
        except ImportError:
            pytest.skip("lightgbm not installed")
        model = get_model("lgb", n_leaves=16)
        assert isinstance(model, LightGBMRanker)
        assert model.n_leaves == 16

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_model("unknown_model")


# ---------------------------------------------------------------------------
# walk_forward_ml.py
# ---------------------------------------------------------------------------

class TestWalkForwardRunner:
    def test_runner_initialises(self):
        model = RidgeFactorModel()
        runner = WalkForwardRunner(model)
        assert runner.model is not None

    def test_result_dataclass(self):
        from datetime import date
        result = WalkForwardResult(
            oos_scores=pd.DataFrame(),
            ic_series=pd.Series(),
            mean_ic=0.05,
            ic_tstat=2.5,
            n_folds=2,
            folds_completed=2,
            folds_failed=0,
        )
        assert result.mean_ic == 0.05
        assert result.folds_completed == 2

    def test_summary_string(self):
        result = WalkForwardResult(
            oos_scores=pd.DataFrame(),
            ic_series=pd.Series(),
            mean_ic=0.05,
            ic_tstat=2.5,
            n_folds=2,
            folds_completed=2,
            folds_failed=0,
        )
        model = RidgeFactorModel()
        runner = WalkForwardRunner(model)
        summary = runner.summary(result)
        assert isinstance(summary, str)
        assert "Walk-Forward" in summary.upper() or "WALK" in summary.upper()
