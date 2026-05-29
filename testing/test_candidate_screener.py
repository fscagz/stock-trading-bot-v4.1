from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from bot.backtest.candidate_screener import CandidateScreener
from bot.config import V4Config

_TARGET_DATE = date(2025, 3, 10)
_PREV_DATE = date(2025, 3, 7)

ASSETS_RESPONSE = [
    {"symbol": "ASTC",    "exchange": "NASDAQ", "status": "active", "tradable": True},
    {"symbol": "SNDL",    "exchange": "NASDAQ", "status": "active", "tradable": True},
    {"symbol": "OTC1",    "exchange": "OTC",    "status": "active", "tradable": True},
    {"symbol": "JOBY.WS", "exchange": "NYSE",   "status": "active", "tradable": True},
]


def _make_df(prev_close: float, day_high: float, day_close: float) -> pd.DataFrame:
    idx = pd.to_datetime([_PREV_DATE, _TARGET_DATE])
    return pd.DataFrame(
        {
            "open": [prev_close, day_close * 0.99],
            "high": [prev_close, day_high],
            "low": [prev_close * 0.99, day_close * 0.98],
            "close": [prev_close, day_close],
            "volume": [1_000_000, 5_000_000],
        },
        index=idx,
    )


def _make_screener(universe: list | None = None) -> CandidateScreener:
    cfg = V4Config()
    screener = CandidateScreener(cfg, "key", "secret")
    if universe is not None:
        screener._universe = universe
    return screener


def test_stage1_filter_includes_5pct_mover():
    # ASTC: high=2.30 vs prev_close=2.00 → 15% gain, close=2.25 → passes
    screener = _make_screener(["ASTC", "SNDL"])
    daily_data = {
        "ASTC": _make_df(2.00, 2.30, 2.25),
        "SNDL": _make_df(1.20, 1.24, 1.22),
    }
    with patch("bot.backtest.candidate_screener.get_daily_batch", return_value=daily_data):
        candidates = screener.candidates_for_date(_TARGET_DATE)
    assert "ASTC" in candidates


def test_stage1_filter_excludes_low_gain():
    # SNDL: 3.3% gain — below stage1_min_price_change_pct (5%)
    screener = _make_screener(["ASTC", "SNDL"])
    daily_data = {
        "ASTC": _make_df(2.00, 2.30, 2.25),
        "SNDL": _make_df(1.20, 1.24, 1.22),
    }
    with patch("bot.backtest.candidate_screener.get_daily_batch", return_value=daily_data):
        candidates = screener.candidates_for_date(_TARGET_DATE)
    assert "SNDL" not in candidates


def test_stage1_filter_excludes_close_below_min_price():
    # 15% gain but close=0.32 — below stage1_min_price (0.50)
    screener = _make_screener(["LOWP"])
    daily_data = {"LOWP": _make_df(0.28, 0.33, 0.32)}
    with patch("bot.backtest.candidate_screener.get_daily_batch", return_value=daily_data):
        candidates = screener.candidates_for_date(_TARGET_DATE)
    assert "LOWP" not in candidates


def test_universe_loaded_from_alpaca_on_first_call():
    screener = _make_screener()  # no pre-loaded universe
    assert screener._universe is None
    with patch("bot.backtest.candidate_screener.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            json=lambda: ASSETS_RESPONSE,
            raise_for_status=lambda: None,
        )
        with patch("bot.backtest.candidate_screener.get_daily_batch", return_value={}):
            screener.candidates_for_date(_TARGET_DATE)
    assert screener._universe is not None
    assert "ASTC" in screener._universe
    assert "OTC1" not in screener._universe
    assert "JOBY.WS" not in screener._universe


def test_batch_failure_does_not_crash():
    screener = _make_screener(["ASTC"])
    with patch("bot.backtest.candidate_screener.get_daily_batch", side_effect=Exception("network")):
        candidates = screener.candidates_for_date(_TARGET_DATE)
    assert candidates == []
