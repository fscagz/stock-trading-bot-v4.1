from unittest.mock import MagicMock, patch
from bot.config import V4Config
from bot.scanner.market_scanner import MarketScanner


MOVERS_RESPONSE = {
    "gainers": [
        {"symbol": "ASTC", "percent_change": 15.3, "price": 2.30, "volume": 5_000_000},
        {"symbol": "SNDL", "percent_change": 3.0, "price": 1.20, "volume": 2_000_000},   # < 5%, filtered
        {"symbol": "GME",  "percent_change": 8.0, "price": 0.30, "volume": 3_000_000},   # < $0.50, filtered
    ]
}

MOST_ACTIVES_RESPONSE = {
    "most_actives": [
        {"symbol": "ASTC", "volume": 5_000_000},
        {"symbol": "NVDA", "volume": 80_000_000},
    ]
}


def _make_scanner():
    cfg = V4Config()
    watchlist = MagicMock()
    return MarketScanner("key", "secret", cfg, watchlist), watchlist


def test_stage1_filter_requires_5pct_gain():
    scanner, watchlist = _make_scanner()
    with patch("bot.scanner.market_scanner.requests.get") as mock_get:
        mock_get.side_effect = [
            MagicMock(json=lambda: MOVERS_RESPONSE, raise_for_status=lambda: None),
            MagicMock(json=lambda: MOST_ACTIVES_RESPONSE, raise_for_status=lambda: None),
        ]
        scanner.scan_once()
    added = {call.args[0] for call in watchlist.add.call_args_list}
    assert "ASTC" in added
    assert "SNDL" not in added


def test_stage1_filter_requires_min_price():
    scanner, watchlist = _make_scanner()
    with patch("bot.scanner.market_scanner.requests.get") as mock_get:
        mock_get.side_effect = [
            MagicMock(json=lambda: MOVERS_RESPONSE, raise_for_status=lambda: None),
            MagicMock(json=lambda: MOST_ACTIVES_RESPONSE, raise_for_status=lambda: None),
        ]
        scanner.scan_once()
    added = {call.args[0] for call in watchlist.add.call_args_list}
    assert "GME" not in added


def test_symbol_on_both_lists_is_high_priority():
    scanner, watchlist = _make_scanner()
    with patch("bot.scanner.market_scanner.requests.get") as mock_get:
        mock_get.side_effect = [
            MagicMock(json=lambda: MOVERS_RESPONSE, raise_for_status=lambda: None),
            MagicMock(json=lambda: MOST_ACTIVES_RESPONSE, raise_for_status=lambda: None),
        ]
        scanner.scan_once()
    # ASTC is on both lists
    calls = {call.args[0]: call.kwargs for call in watchlist.add.call_args_list}
    assert calls.get("ASTC", {}).get("high_priority") is True
