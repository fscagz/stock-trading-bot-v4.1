import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from bot.backtest.bar_fetcher import BarFetcher

_SAMPLE_BARS_RAW = [
    # Regular session bars: EDT=UTC-4, so 9:30 AM EDT = 13:30 UTC
    {"t": "2025-03-10T13:30:00Z", "o": 2.0, "h": 2.5, "l": 1.9, "c": 2.3, "v": 50000},
    {"t": "2025-03-10T13:31:00Z", "o": 2.3, "h": 2.6, "l": 2.2, "c": 2.5, "v": 40000},
    # Pre-market bar: 9:00 AM EDT = 13:00 UTC — should be filtered out
    {"t": "2025-03-10T13:00:00Z", "o": 2.0, "h": 2.1, "l": 1.95, "c": 2.05, "v": 5000},
]


def _fetcher(tmpdir: str) -> BarFetcher:
    return BarFetcher("key", "secret", cache_dir=tmpdir)


def test_cache_miss_fetches_api_and_writes_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        fetcher = _fetcher(tmpdir)
        resp = MagicMock(raise_for_status=lambda: None)
        resp.json.return_value = {"bars": _SAMPLE_BARS_RAW, "next_page_token": None}
        with patch("bot.backtest.bar_fetcher.requests.get", return_value=resp):
            bars = fetcher.fetch("ASTC", date(2025, 3, 10))
        assert len(bars) == 2  # pre-market bar filtered out
        assert (Path(tmpdir) / "ASTC_2025-03-10.json").exists()


def test_cache_hit_skips_api():
    with tempfile.TemporaryDirectory() as tmpdir:
        fetcher = _fetcher(tmpdir)
        cached = [{"t": "2025-03-10T13:30:00+00:00", "o": 2.0, "h": 2.5, "l": 1.9, "c": 2.3, "v": 50000}]
        (Path(tmpdir) / "ASTC_2025-03-10.json").write_text(json.dumps(cached))
        with patch("bot.backtest.bar_fetcher.requests.get") as mock_get:
            bars = fetcher.fetch("ASTC", date(2025, 3, 10))
        mock_get.assert_not_called()
        assert len(bars) == 1
        assert bars[0].symbol == "ASTC"
        assert bars[0].close == 2.3


def test_session_filter_excludes_premarket():
    with tempfile.TemporaryDirectory() as tmpdir:
        fetcher = _fetcher(tmpdir)
        resp = MagicMock(raise_for_status=lambda: None)
        resp.json.return_value = {"bars": _SAMPLE_BARS_RAW, "next_page_token": None}
        with patch("bot.backtest.bar_fetcher.requests.get", return_value=resp):
            bars = fetcher.fetch("ASTC", date(2025, 3, 10))
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        for b in bars:
            ts_et = b.timestamp.astimezone(ET)
            assert (ts_et.hour, ts_et.minute) >= (9, 30)


def test_pagination_collects_all_bars():
    with tempfile.TemporaryDirectory() as tmpdir:
        fetcher = _fetcher(tmpdir)
        resp1 = MagicMock(raise_for_status=lambda: None)
        resp1.json.return_value = {"bars": [_SAMPLE_BARS_RAW[0]], "next_page_token": "tok123"}
        resp2 = MagicMock(raise_for_status=lambda: None)
        resp2.json.return_value = {"bars": [_SAMPLE_BARS_RAW[1]], "next_page_token": None}
        with patch("bot.backtest.bar_fetcher.requests.get") as mock_get:
            mock_get.side_effect = [resp1, resp2]
            bars = fetcher.fetch("ASTC", date(2025, 3, 10))
        assert len(bars) == 2
        assert mock_get.call_count == 2


def test_api_error_returns_empty_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        fetcher = _fetcher(tmpdir)
        with patch("bot.backtest.bar_fetcher.requests.get", side_effect=Exception("network")):
            bars = fetcher.fetch("ASTC", date(2025, 3, 10))
        assert bars == []
