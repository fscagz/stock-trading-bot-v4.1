"""
Walk-forward fold generator for ML model training.

Produces rolling (train, test) date splits with strict date discipline:
  - No training label uses any price data that falls in the test period
  - Minimum training window enforced before first out-of-sample fold
  - No overlap between folds

This module provides the date splits. Actual model training lives in
bot/models/walk_forward_ml.py (Phase 7). The backtest engine uses point-in-time
data loading to achieve the same discipline for the baseline factor model —
walk_forward.py is specifically for generating ML training windows.

Usage
-----
    from backtest.walk_forward import generate_folds, FoldSpec

    folds = generate_folds(
        start="2010-01-01",
        end="2023-12-31",
        train_years=3,
        test_months=6,
        min_train_years=5,
    )
    for fold in folds:
        print(fold.train_start, fold.train_end, "→", fold.test_start, fold.test_end)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterator, List

import pandas as pd

from data.integrity_checks import assert_no_train_test_overlap


@dataclass(frozen=True)
class FoldSpec:
    """One walk-forward fold: a training window and a test window."""
    fold_index: int
    train_start: date
    train_end: date       # last date INCLUDED in training
    test_start: date      # first date of the test (out-of-sample) period
    test_end: date        # last date of the test period

    def __post_init__(self) -> None:
        # Hard assertion: train must end strictly before test starts
        assert_no_train_test_overlap(
            self.train_end, self.test_start,
            context=f"fold_{self.fold_index}"
        )

    @property
    def train_days(self) -> int:
        return (self.train_end - self.train_start).days

    @property
    def test_days(self) -> int:
        return (self.test_end - self.test_start).days


def generate_folds(
    start: str,
    end: str,
    train_years: int = 3,
    test_months: int = 6,
    min_train_years: int = 5,
    step_months: int = 6,
) -> List[FoldSpec]:
    """
    Generate walk-forward (train, test) splits.

    The first fold that has >= min_train_years of history before its test
    period is the first out-of-sample fold. All prior history is used
    as expanding training data for that fold.

    Parameters
    ----------
    start : str
        First available date in the dataset ('YYYY-MM-DD').
    end : str
        Last available date ('YYYY-MM-DD').
    train_years : int
        Minimum rolling training window length in years.
    test_months : int
        Length of each test (out-of-sample) period in months.
    min_train_years : int
        Minimum total history required before generating the first fold.
        E.g. 5 means the first fold won't be generated until 5 years of
        data are available. This prevents over-fitting on short histories.
    step_months : int
        How far to advance the test window between folds.

    Returns
    -------
    list of FoldSpec
        Sorted by fold_index (chronological).
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # First test period start: start + min_train_years
    first_test_start = start_ts + pd.DateOffset(years=min_train_years)
    if first_test_start >= end_ts:
        raise ValueError(
            f"Not enough data: min_train_years={min_train_years} requires data through "
            f"{first_test_start.date()}, but end={end}."
        )

    folds: List[FoldSpec] = []
    test_start_ts = first_test_start
    fold_idx = 0

    while test_start_ts < end_ts:
        test_end_ts = min(test_start_ts + pd.DateOffset(months=test_months), end_ts)

        # Training window: fixed length ending just before test_start
        train_end_ts = test_start_ts - pd.Timedelta(days=1)
        train_start_ts = train_end_ts - pd.DateOffset(years=train_years)
        # Never start before the dataset start
        train_start_ts = max(train_start_ts, start_ts)

        fold = FoldSpec(
            fold_index=fold_idx,
            train_start=train_start_ts.date(),
            train_end=train_end_ts.date(),
            test_start=test_start_ts.date(),
            test_end=test_end_ts.date(),
        )
        folds.append(fold)

        test_start_ts += pd.DateOffset(months=step_months)
        fold_idx += 1

    return folds


def expanding_folds(
    start: str,
    end: str,
    test_months: int = 6,
    min_train_years: int = 5,
    step_months: int = 6,
) -> List[FoldSpec]:
    """
    Like generate_folds but uses an expanding (not rolling) training window.
    The training window always starts at `start` and grows with each fold.

    Expanding windows are less prone to forgetting old regimes but can give
    older data too much influence. Compare with rolling folds.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    first_test_start = start_ts + pd.DateOffset(years=min_train_years)
    if first_test_start >= end_ts:
        raise ValueError(
            f"Not enough data for min_train_years={min_train_years}."
        )

    folds: List[FoldSpec] = []
    test_start_ts = first_test_start
    fold_idx = 0

    while test_start_ts < end_ts:
        test_end_ts = min(test_start_ts + pd.DateOffset(months=test_months), end_ts)
        train_end_ts = test_start_ts - pd.Timedelta(days=1)

        fold = FoldSpec(
            fold_index=fold_idx,
            train_start=start_ts.date(),
            train_end=train_end_ts.date(),
            test_start=test_start_ts.date(),
            test_end=test_end_ts.date(),
        )
        folds.append(fold)
        test_start_ts += pd.DateOffset(months=step_months)
        fold_idx += 1

    return folds


def describe_folds(folds: List[FoldSpec]) -> str:
    """Return a human-readable summary of all folds."""
    lines = [f"{'Fold':<6} {'Train start':<14} {'Train end':<14} {'Test start':<14} {'Test end':<14}"]
    lines.append("-" * 65)
    for f in folds:
        lines.append(
            f"{f.fold_index:<6} {str(f.train_start):<14} {str(f.train_end):<14} "
            f"{str(f.test_start):<14} {str(f.test_end):<14}"
        )
    lines.append(f"\n{len(folds)} folds total.")
    return "\n".join(lines)
