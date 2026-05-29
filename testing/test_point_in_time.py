# test_point_in_time.py
# Run from bot/:  pytest ../testing/test_point_in_time.py -v

import sys
from pathlib import Path

bot_dir = Path(__file__).resolve().parent.parent / "bot"
if str(bot_dir) not in sys.path:
    sys.path.insert(0, str(bot_dir))

import pandas as pd
from data.point_in_time import (
    slice_daily_as_of,
    get_daily_as_of,
    get_daily_batch_as_of,
    rebalance_dates,
)


def test_slice_daily_as_of_excludes_as_of_date_by_default():
    df = pd.DataFrame(
        {"close": [98, 99, 100, 101]},
        index=pd.to_datetime(["2024-01-10", "2024-01-11", "2024-01-12", "2024-01-15"]),
    )
    out = slice_daily_as_of(df, "2024-01-15", include_as_of_date=False)
    assert len(out) == 3
    assert out.index.max() < pd.Timestamp("2024-01-15")
    assert pd.Timestamp("2024-01-12") in out.index


def test_slice_daily_as_of_includes_as_of_date_when_asked():
    df = pd.DataFrame(
        {"close": [98, 99, 100]},
        index=pd.to_datetime(["2024-01-10", "2024-01-11", "2024-01-12"]),
    )
    out = slice_daily_as_of(df, "2024-01-12", include_as_of_date=True)
    assert len(out) == 3
    assert pd.Timestamp("2024-01-12") in out.index


def test_slice_daily_as_of_empty():
    df = pd.DataFrame(columns=["close"])
    out = slice_daily_as_of(df, "2024-01-15")
    assert out.empty


def test_rebalance_dates():
    dr = rebalance_dates("2024-01-01", "2024-06-30", freq="ME")
    assert len(dr) >= 6
    assert dr[0] <= pd.Timestamp("2024-01-31")
    assert dr[-1] <= pd.Timestamp("2024-06-30")


def test_get_daily_as_of_returns_data():
    # Integration: fetch AAPL as of a fixed past date
    df = get_daily_as_of("AAPL", "2024-01-15", lookback_period="1mo")
    assert isinstance(df, pd.DataFrame)
    assert "close" in df.columns
    if not df.empty:
        assert df.index.max() < pd.Timestamp("2024-01-15")


def test_get_daily_batch_as_of_returns_dict():
    panel = get_daily_batch_as_of(["AAPL", "MSFT"], "2024-01-15", lookback_period="1mo")
    assert isinstance(panel, dict)
    assert "AAPL" in panel
    assert "MSFT" in panel
    for sym, df in panel.items():
        if not df.empty:
            assert df.index.max() < pd.Timestamp("2024-01-15")
