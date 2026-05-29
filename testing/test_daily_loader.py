# test_daily_loader.py
# Run from bot/:  python -m pytest ../testing/test_daily_loader.py -v
# Or from project root with PYTHONPATH=bot:  pytest testing/test_daily_loader.py -v

import sys
from pathlib import Path

# Allow importing from bot when run from project root
bot_dir = Path(__file__).resolve().parent.parent / "bot"
if str(bot_dir) not in sys.path:
    sys.path.insert(0, str(bot_dir))

from data.daily_loader import get_daily, get_daily_batch, OHLCV_COLS


def test_get_daily_returns_dataframe():
    df = get_daily("AAPL", period="1mo")
    assert not df.empty
    for c in ["open", "high", "low", "close", "volume"]:
        assert c in df.columns
    assert df.index.name == "date"


def test_get_daily_empty_with_bad_symbol():
    df = get_daily("NOTAREALTICKER123", period="1mo")
    assert df.empty
    assert list(df.columns) == OHLCV_COLS or df.columns.tolist() == []


def test_get_daily_batch_returns_dict():
    batch = get_daily_batch(["AAPL", "MSFT"], period="1mo")
    assert isinstance(batch, dict)
    assert "AAPL" in batch
    assert "MSFT" in batch
    assert not batch["AAPL"].empty
    assert list(batch["AAPL"].columns) == ["open", "high", "low", "close", "volume"] or set(batch["AAPL"].columns) >= {"open", "high", "low", "close", "volume"}


def test_get_daily_batch_empty_list():
    assert get_daily_batch([]) == {}
