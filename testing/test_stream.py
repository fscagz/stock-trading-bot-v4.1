from bot.intraday.data.stream import BarStream


def test_subscribe_adds_symbol():
    stream = BarStream("127.0.0.1", 4001, 1, [])
    stream.subscribe("ASTC")
    assert "ASTC" in stream.symbols


def test_unsubscribe_removes_symbol():
    stream = BarStream("127.0.0.1", 4001, 1, ["ASTC"])
    stream.unsubscribe("ASTC")
    assert "ASTC" not in stream.symbols


def test_subscribe_is_idempotent():
    stream = BarStream("127.0.0.1", 4001, 1, [])
    stream.subscribe("ASTC")
    stream.subscribe("ASTC")
    assert len([s for s in stream.symbols if s == "ASTC"]) == 1
