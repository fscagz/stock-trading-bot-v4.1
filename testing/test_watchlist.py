from unittest.mock import MagicMock, patch
from bot.config import V4Config
from bot.scanner.watchlist import Watchlist


def _make_watchlist():
    cfg = V4Config()
    stream = MagicMock()
    watchlist = Watchlist(stream, cfg)
    return watchlist, stream


def test_add_subscribes_to_stream():
    watchlist, stream = _make_watchlist()
    with patch.object(watchlist, "_load_baseline_volume", return_value=1000.0):
        watchlist.add("ASTC")
    stream.subscribe.assert_called_once_with("ASTC")


def test_add_is_idempotent():
    watchlist, stream = _make_watchlist()
    with patch.object(watchlist, "_load_baseline_volume", return_value=1000.0):
        watchlist.add("ASTC")
        watchlist.add("ASTC")
    assert stream.subscribe.call_count == 1


def test_remove_unsubscribes_from_stream():
    watchlist, stream = _make_watchlist()
    with patch.object(watchlist, "_load_baseline_volume", return_value=1000.0):
        watchlist.add("ASTC")
    watchlist.remove("ASTC")
    stream.unsubscribe.assert_called_once_with("ASTC")


def test_get_baseline_volume_returns_stored_value():
    watchlist, stream = _make_watchlist()
    with patch.object(watchlist, "_load_baseline_volume", return_value=2500.0):
        watchlist.add("ASTC")
    assert watchlist.get_baseline_volume("ASTC") == 2500.0


def test_get_baseline_volume_unknown_symbol_returns_none():
    watchlist, stream = _make_watchlist()
    assert watchlist.get_baseline_volume("UNKNOWN") is None


def test_symbols_property_returns_current_set():
    watchlist, stream = _make_watchlist()
    with patch.object(watchlist, "_load_baseline_volume", return_value=1000.0):
        watchlist.add("ASTC")
        watchlist.add("NVDA")
    assert "ASTC" in watchlist.symbols
    assert "NVDA" in watchlist.symbols
