# test_store.py
# Run from bot/:  pytest ../testing/test_store.py -v

import sys
import tempfile
from pathlib import Path

bot_dir = Path(__file__).resolve().parent.parent / "bot"
if str(bot_dir) not in sys.path:
    sys.path.insert(0, str(bot_dir))

import pandas as pd
import data.store as store_module


def test_save_and_load_daily():
    with tempfile.TemporaryDirectory() as tmp:
        store_module.CACHE_DIR = Path(tmp)
        df = pd.DataFrame(
            {"open": [100], "high": [101], "low": [99], "close": [100.5], "volume": [1e6]},
            index=pd.to_datetime(["2024-01-15"]),
        )
        df.index.name = "date"
        store_module.save_daily("TEST", df)
        loaded = store_module.load_daily("TEST")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded.index.name == "date"
        assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
        assert loaded["close"].iloc[0] == 100.5


def test_load_daily_missing_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        store_module.CACHE_DIR = Path(tmp)
        assert store_module.load_daily("NOMATCHXYZ") is None


def test_save_and_load_fundamentals():
    with tempfile.TemporaryDirectory() as tmp:
        store_module.CACHE_DIR = Path(tmp)
        data = {"pe_ratio": 25.0, "roe": 0.15, "revenue_growth": None}
        store_module.save_fundamentals("TEST", data)
        loaded = store_module.load_fundamentals("TEST")
        assert loaded is not None
        assert loaded["pe_ratio"] == 25.0
        assert loaded["roe"] == 0.15
        assert loaded["revenue_growth"] is None


def test_load_fundamentals_missing_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        store_module.CACHE_DIR = Path(tmp)
        assert store_module.load_fundamentals("NOMATCHXYZ") is None


def test_is_daily_stale():
    with tempfile.TemporaryDirectory() as tmp:
        store_module.CACHE_DIR = Path(tmp)
        assert store_module.is_daily_stale("MISSING", max_age_days=1.0) is True
        df = pd.DataFrame(
            {"open": [100], "high": [101], "low": [99], "close": [100], "volume": [1e6]},
            index=pd.to_datetime(["2024-01-15"]),
        )
        df.index.name = "date"
        store_module.save_daily("FRESH", df)
        assert store_module.is_daily_stale("FRESH", max_age_days=1.0) is False
        assert store_module.is_daily_stale("FRESH", max_age_days=0.0) is True


def test_is_fundamentals_stale():
    with tempfile.TemporaryDirectory() as tmp:
        store_module.CACHE_DIR = Path(tmp)
        assert store_module.is_fundamentals_stale("MISSING", max_age_days=7.0) is True
        store_module.save_fundamentals("FRESH", {"pe_ratio": 20.0})
        assert store_module.is_fundamentals_stale("FRESH", max_age_days=7.0) is False
