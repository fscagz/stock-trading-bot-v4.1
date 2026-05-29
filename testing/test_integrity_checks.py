# test_integrity_checks.py
# Run from bot/:  pytest ../testing/test_integrity_checks.py -v

import sys
from datetime import date
from pathlib import Path

bot_dir = Path(__file__).resolve().parent.parent / "bot"
if str(bot_dir) not in sys.path:
    sys.path.insert(0, str(bot_dir))

import pandas as pd
import pytest
from data.integrity_checks import (
    IntegrityError,
    assert_no_lookahead,
    assert_no_train_test_overlap,
    assert_universe_snapshot_used,
    check_lookahead_batch,
    check_lookahead_single,
    run_preflight_checks,
    scan_feature_matrix_for_lookahead,
    survivorship_bias_report,
)


# ---------------------------------------------------------------------------
# 1. Look-ahead: single check
# ---------------------------------------------------------------------------

def test_no_lookahead_when_filing_before_rebalance():
    assert check_lookahead_single("AAPL", date(2022, 6, 30), date(2022, 5, 1)) is None


def test_no_lookahead_when_filing_on_rebalance_day():
    # Same-day filing is fine — market could have seen it
    assert check_lookahead_single("AAPL", date(2022, 6, 30), date(2022, 6, 30)) is None


def test_lookahead_detected_when_filing_after_rebalance():
    msg = check_lookahead_single("AAPL", date(2022, 6, 30), date(2022, 7, 15))
    assert msg is not None
    assert "LOOK-AHEAD BIAS" in msg
    assert "AAPL" in msg


def test_no_lookahead_when_filing_date_is_none():
    # None filing date means no fundamental data used — no look-ahead risk
    assert check_lookahead_single("AAPL", date(2022, 6, 30), None) is None


# ---------------------------------------------------------------------------
# 2. Look-ahead: batch check
# ---------------------------------------------------------------------------

def test_batch_clean():
    violations = check_lookahead_batch(
        date(2022, 6, 30),
        {"AAPL": date(2022, 5, 1), "MSFT": date(2022, 4, 15)},
    )
    assert violations == []


def test_batch_detects_violations():
    violations = check_lookahead_batch(
        date(2022, 6, 30),
        {"AAPL": date(2022, 5, 1), "MSFT": date(2022, 8, 1)},  # MSFT is future
    )
    assert len(violations) == 1
    assert "MSFT" in violations[0]


def test_batch_multiple_violations():
    violations = check_lookahead_batch(
        date(2022, 6, 30),
        {"A": date(2022, 7, 1), "B": date(2022, 8, 1), "C": date(2022, 5, 1)},
    )
    assert len(violations) == 2


# ---------------------------------------------------------------------------
# 3. assert_no_lookahead: raises on violation
# ---------------------------------------------------------------------------

def test_assert_no_lookahead_clean():
    # Should not raise
    assert_no_lookahead(
        date(2022, 6, 30),
        {"AAPL": date(2022, 5, 1), "MSFT": None},
    )


def test_assert_no_lookahead_raises():
    with pytest.raises(IntegrityError, match="LOOK-AHEAD BIAS"):
        assert_no_lookahead(
            date(2022, 6, 30),
            {"AAPL": date(2022, 7, 15)},
        )


def test_assert_no_lookahead_includes_context():
    with pytest.raises(IntegrityError, match="my_backtest"):
        assert_no_lookahead(
            date(2022, 6, 30),
            {"AAPL": date(2022, 7, 15)},
            context="my_backtest",
        )


# ---------------------------------------------------------------------------
# 4. scan_feature_matrix_for_lookahead
# ---------------------------------------------------------------------------

def test_scan_feature_matrix_clean():
    df = pd.DataFrame({
        "momentum_3m": [0.1, 0.2],
        "filing_date": [date(2022, 5, 1), date(2022, 4, 15)],
    }, index=["AAPL", "MSFT"])
    violations = scan_feature_matrix_for_lookahead(df, date(2022, 6, 30))
    assert violations == []


def test_scan_feature_matrix_detects_violation():
    df = pd.DataFrame({
        "momentum_3m": [0.1, 0.2],
        "filing_date": [date(2022, 5, 1), date(2022, 8, 1)],
    }, index=["AAPL", "MSFT"])
    violations = scan_feature_matrix_for_lookahead(df, date(2022, 6, 30))
    assert len(violations) == 1
    assert "MSFT" in violations[0]


def test_scan_feature_matrix_no_filing_col():
    df = pd.DataFrame({"momentum_3m": [0.1]}, index=["AAPL"])
    violations = scan_feature_matrix_for_lookahead(df, date(2022, 6, 30))
    assert violations == []


# ---------------------------------------------------------------------------
# 5. Survivorship bias report
# ---------------------------------------------------------------------------

def test_survivorship_no_bias():
    report = survivorship_bias_report(
        historical_snapshot=["AAPL", "MSFT", "GOOG"],
        current_tickers=["AAPL", "MSFT", "GOOG"],
        snapshot_date=date(2015, 1, 1),
    )
    assert report["n_biased_additions"] == 0
    assert report["biased_additions"] == []
    assert "No survivorship bias" in report["summary"]


def test_survivorship_detects_additions():
    report = survivorship_bias_report(
        historical_snapshot=["AAPL", "MSFT"],
        current_tickers=["AAPL", "MSFT", "NVDA", "META"],  # NVDA and META added later
        snapshot_date=date(2015, 1, 1),
    )
    assert report["n_biased_additions"] == 2
    assert "NVDA" in report["biased_additions"]
    assert "META" in report["biased_additions"]
    assert report["biased_pct"] == 0.5


def test_survivorship_detects_delisted():
    report = survivorship_bias_report(
        historical_snapshot=["AAPL", "MSFT", "LEHMAN"],  # LEHMAN went bankrupt
        current_tickers=["AAPL", "MSFT"],
        snapshot_date=date(2007, 1, 1),
    )
    assert "LEHMAN" in report["delisted_or_removed"]
    assert report["n_delisted_or_removed"] == 1


def test_survivorship_report_includes_snapshot_date():
    report = survivorship_bias_report([], [], snapshot_date=date(2015, 6, 30))
    assert report["snapshot_date"] == date(2015, 6, 30)


# ---------------------------------------------------------------------------
# 6. assert_universe_snapshot_used
# ---------------------------------------------------------------------------

def test_snapshot_used_clean():
    # All backtest tickers are in snapshot → no error
    assert_universe_snapshot_used(
        backtest_tickers=["AAPL", "MSFT"],
        snapshot_tickers=["AAPL", "MSFT", "GOOG"],
        rebalance_date=date(2015, 1, 31),
    )


def test_snapshot_used_raises_on_extra():
    with pytest.raises(IntegrityError, match="(?i)survivorship bias"):
        assert_universe_snapshot_used(
            backtest_tickers=["AAPL", "MSFT", "NVDA"],  # NVDA not in snapshot
            snapshot_tickers=["AAPL", "MSFT"],
            rebalance_date=date(2015, 1, 31),
        )


def test_snapshot_used_tolerance():
    # max_extra_pct=0.1 allows up to 10% extra tickers
    assert_universe_snapshot_used(
        backtest_tickers=["AAPL", "MSFT", "NVDA"],  # 1/2 = 50% extra → exceeds 10%
        snapshot_tickers=["AAPL", "MSFT"],
        rebalance_date=date(2015, 1, 31),
        max_extra_pct=0.6,  # 60% tolerance → passes
    )


# ---------------------------------------------------------------------------
# 7. assert_no_train_test_overlap
# ---------------------------------------------------------------------------

def test_train_test_no_overlap():
    assert_no_train_test_overlap(date(2020, 12, 31), date(2021, 1, 1))


def test_train_test_same_day_raises():
    with pytest.raises(IntegrityError, match="Train/test overlap"):
        assert_no_train_test_overlap(date(2021, 1, 1), date(2021, 1, 1))


def test_train_test_overlap_raises():
    with pytest.raises(IntegrityError, match="Train/test overlap"):
        assert_no_train_test_overlap(date(2021, 6, 30), date(2021, 3, 31))


def test_train_test_includes_context():
    with pytest.raises(IntegrityError, match="fold_3"):
        assert_no_train_test_overlap(date(2021, 1, 1), date(2020, 12, 31), context="fold_3")


# ---------------------------------------------------------------------------
# 8. run_preflight_checks (composite gate)
# ---------------------------------------------------------------------------

def test_preflight_passes():
    run_preflight_checks(
        rebalance_date=date(2022, 6, 30),
        filing_dates={"AAPL": date(2022, 5, 1), "MSFT": None},
        backtest_tickers=["AAPL"],
        snapshot_tickers=["AAPL", "MSFT"],
    )


def test_preflight_fails_on_lookahead():
    with pytest.raises(IntegrityError, match="LOOK-AHEAD BIAS"):
        run_preflight_checks(
            rebalance_date=date(2022, 6, 30),
            filing_dates={"AAPL": date(2022, 8, 1)},
            backtest_tickers=["AAPL"],
            snapshot_tickers=["AAPL"],
        )


def test_preflight_fails_on_survivorship():
    with pytest.raises(IntegrityError, match="(?i)survivorship bias"):
        run_preflight_checks(
            rebalance_date=date(2022, 6, 30),
            filing_dates={"AAPL": date(2022, 5, 1), "NVDA": date(2022, 5, 1)},
            backtest_tickers=["AAPL", "NVDA"],  # NVDA not in snapshot
            snapshot_tickers=["AAPL"],
        )
