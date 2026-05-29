from datetime import date
from unittest.mock import MagicMock, patch
from bot.config import V4Config
from bot.scanner.market_scanner import MarketScanner


# Alpaca snapshot format: dailyBar.c = current price, prevDailyBar.c = previous close
SNAPSHOT_RESPONSE = {
    # ASTC: 15% gain, $2.30 — passes Stage 1
    "ASTC": {"dailyBar": {"c": 2.30, "v": 5_000_000}, "prevDailyBar": {"c": 2.00}},
    # SNDL: 3.3% gain — below 5% threshold, filtered
    "SNDL": {"dailyBar": {"c": 1.24, "v": 2_000_000}, "prevDailyBar": {"c": 1.20}},
    # GME: 8% gain but price $0.324 — below min_price, filtered
    "GME":  {"dailyBar": {"c": 0.324, "v": 3_000_000}, "prevDailyBar": {"c": 0.30}},
}

ASSETS_RESPONSE = [
    {"symbol": "ASTC",    "exchange": "NASDAQ", "status": "active", "tradable": True},
    {"symbol": "SNDL",    "exchange": "NASDAQ", "status": "active", "tradable": True},
    {"symbol": "GME",     "exchange": "NYSE",   "status": "active", "tradable": True},
    {"symbol": "OTC1",    "exchange": "OTC",    "status": "active", "tradable": True},
    {"symbol": "JOBY.WS", "exchange": "NYSE",   "status": "active", "tradable": True},  # warrant
    {"symbol": "XYZ.R",   "exchange": "NASDAQ", "status": "active", "tradable": True},  # right
]


def _make_scanner(prepopulate_universe=True):
    cfg = V4Config()
    watchlist = MagicMock()
    scanner = MarketScanner("key", "secret", cfg, watchlist)
    if prepopulate_universe:
        scanner._universe = ["ASTC", "SNDL", "GME"]
        scanner._universe_date = date.today()
    return scanner, watchlist


def _mock_snapshot_get(mock_get):
    mock_get.return_value = MagicMock(
        json=lambda: SNAPSHOT_RESPONSE,
        raise_for_status=lambda: None,
    )


def test_stage1_filter_requires_5pct_gain():
    scanner, watchlist = _make_scanner()
    with patch("bot.scanner.market_scanner.requests.get") as mock_get:
        _mock_snapshot_get(mock_get)
        scanner.scan_once()
    added = {call.args[0] for call in watchlist.add.call_args_list}
    assert "ASTC" in added
    assert "SNDL" not in added


def test_stage1_filter_requires_min_price():
    scanner, watchlist = _make_scanner()
    with patch("bot.scanner.market_scanner.requests.get") as mock_get:
        _mock_snapshot_get(mock_get)
        scanner.scan_once()
    added = {call.args[0] for call in watchlist.add.call_args_list}
    assert "GME" not in added


def test_universe_filters_out_non_common_stock():
    scanner, watchlist = _make_scanner(prepopulate_universe=False)
    with patch("bot.scanner.market_scanner.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            json=lambda: ASSETS_RESPONSE,
            raise_for_status=lambda: None,
        )
        universe = scanner._load_universe()
    assert "ASTC" in universe
    assert "OTC1" not in universe    # wrong exchange
    assert "JOBY.WS" not in universe  # warrant
    assert "XYZ.R" not in universe    # right


def test_universe_refreshes_when_date_changes():
    scanner, watchlist = _make_scanner()
    scanner._universe_date = date(2000, 1, 1)  # stale date
    with patch("bot.scanner.market_scanner.requests.get") as mock_get:
        mock_get.side_effect = [
            MagicMock(json=lambda: ASSETS_RESPONSE, raise_for_status=lambda: None),
            MagicMock(json=lambda: SNAPSHOT_RESPONSE, raise_for_status=lambda: None),
        ]
        scanner.scan_once()
    assert scanner._universe_date == date.today()


def test_failed_snapshot_batch_does_not_crash():
    scanner, watchlist = _make_scanner()
    with patch("bot.scanner.market_scanner.requests.get") as mock_get:
        mock_get.side_effect = Exception("network error")
        scanner.scan_once()  # should not raise
    watchlist.add.assert_not_called()
