from __future__ import annotations
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import List

import pytest

from bot.intraday.data.aggregator import MinuteBarAggregator
from bot.intraday.types import Bar

_UTC = timezone.utc


def _make_5s(symbol: str, ts: datetime, o: float, h: float, l: float, c: float, vol: int):
    """Create a fake ib_insync RealTimeBar-like object."""
    return SimpleNamespace(
        time=ts,
        open_=o,
        high=h,
        low=l,
        close=c,
        volume=vol,
        contract=SimpleNamespace(symbol=symbol),
    )


def _ts(h: int, m: int, s: int) -> datetime:
    return datetime(2024, 1, 15, h, m, s, tzinfo=_UTC)


class TestMinuteBarAggregator:
    def test_no_emission_within_same_minute(self):
        emitted: List[Bar] = []
        agg = MinuteBarAggregator(emitted.append)

        agg.push("AAPL", _make_5s("AAPL", _ts(14, 30, 0), 150.0, 151.0, 149.5, 150.5, 1000))
        agg.push("AAPL", _make_5s("AAPL", _ts(14, 30, 5), 150.5, 152.0, 150.0, 151.0, 800))

        assert emitted == [], "should not emit mid-minute"

    def test_emits_on_minute_boundary(self):
        emitted: List[Bar] = []
        agg = MinuteBarAggregator(emitted.append)

        agg.push("AAPL", _make_5s("AAPL", _ts(14, 30, 0), 150.0, 151.0, 149.5, 150.5, 1000))
        agg.push("AAPL", _make_5s("AAPL", _ts(14, 30, 5), 150.5, 152.0, 150.0, 151.0,  800))
        agg.push("AAPL", _make_5s("AAPL", _ts(14, 31, 0), 151.0, 153.0, 150.8, 152.0,  600))

        assert len(emitted) == 1
        bar = emitted[0]
        assert bar.symbol == "AAPL"
        assert bar.open == pytest.approx(150.0)
        assert bar.high == pytest.approx(152.0)
        assert bar.low == pytest.approx(149.5)
        assert bar.close == pytest.approx(151.0)
        assert bar.volume == 1800
        assert bar.timestamp == _ts(14, 30, 0)

    def test_ohlcv_aggregation_correctness(self):
        emitted: List[Bar] = []
        agg = MinuteBarAggregator(emitted.append)

        agg.push("TSLA", _make_5s("TSLA", _ts(14, 30,  0), 200.0, 205.0, 199.0, 202.0, 500))
        agg.push("TSLA", _make_5s("TSLA", _ts(14, 30,  5), 202.0, 210.0, 201.0, 209.0, 700))
        agg.push("TSLA", _make_5s("TSLA", _ts(14, 30, 10), 209.0, 209.5, 195.0, 196.0, 300))
        agg.push("TSLA", _make_5s("TSLA", _ts(14, 31,  0), 196.0, 197.0, 195.5, 196.5, 200))

        assert len(emitted) == 1
        bar = emitted[0]
        assert bar.open == pytest.approx(200.0)   # first bar's open
        assert bar.high == pytest.approx(210.0)   # max high
        assert bar.low  == pytest.approx(195.0)   # min low
        assert bar.close == pytest.approx(196.0)  # last bar's close
        assert bar.volume == 1500                  # sum

    def test_independent_per_symbol(self):
        emitted: List[Bar] = []
        agg = MinuteBarAggregator(emitted.append)

        agg.push("AAPL", _make_5s("AAPL", _ts(14, 30, 0), 150.0, 151.0, 149.5, 150.5, 100))
        agg.push("TSLA", _make_5s("TSLA", _ts(14, 30, 0), 200.0, 201.0, 199.0, 200.5, 200))

        # Trigger AAPL minute boundary — should only emit AAPL bar
        agg.push("AAPL", _make_5s("AAPL", _ts(14, 31, 0), 150.5, 152.0, 150.0, 151.0, 50))
        assert len(emitted) == 1
        assert emitted[0].symbol == "AAPL"

        # Trigger TSLA minute boundary
        agg.push("TSLA", _make_5s("TSLA", _ts(14, 31, 0), 200.5, 202.0, 200.0, 201.0, 100))
        assert len(emitted) == 2
        assert emitted[1].symbol == "TSLA"

    def test_timestamp_is_minute_start(self):
        emitted: List[Bar] = []
        agg = MinuteBarAggregator(emitted.append)

        agg.push("AAPL", _make_5s("AAPL", _ts(14, 30, 5), 150.0, 151.0, 149.5, 150.5, 100))
        agg.push("AAPL", _make_5s("AAPL", _ts(14, 31, 0), 150.5, 151.0, 150.0, 150.8, 50))

        assert emitted[0].timestamp == _ts(14, 30, 0)  # truncated to minute start
