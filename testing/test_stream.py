from unittest.mock import MagicMock, patch
from bot.intraday.data.stream import BarStream


def test_subscribe_adds_symbol():
    stream = BarStream("key", "secret", [])
    with patch("bot.intraday.data.stream.StockDataStream"):
        mock_client = MagicMock()
        stream._client = mock_client
        stream.subscribe("ASTC")
        assert "ASTC" in stream.symbols


def test_unsubscribe_removes_symbol():
    stream = BarStream("key", "secret", ["ASTC"])
    with patch("bot.intraday.data.stream.StockDataStream"):
        mock_client = MagicMock()
        stream._client = mock_client
        stream.unsubscribe("ASTC")
        assert "ASTC" not in stream.symbols


def test_subscribe_is_idempotent():
    stream = BarStream("key", "secret", [])
    with patch("bot.intraday.data.stream.StockDataStream"):
        mock_client = MagicMock()
        stream._client = mock_client
        stream.subscribe("ASTC")
        stream.subscribe("ASTC")
        assert len([s for s in stream.symbols if s == "ASTC"]) == 1
