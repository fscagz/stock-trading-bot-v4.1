from __future__ import annotations
import json
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

from bot.intraday.types import Bar

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
_PACING_INTERVAL = 10.0  # seconds between IBKR historical data requests

try:
    from ib_insync import IB, Stock
    _HAVE_IBKR = True
except ImportError:
    IB = None  # type: ignore[assignment,misc]
    _HAVE_IBKR = False


class BarFetcher:
    """Fetches 1-minute OHLCV bars from IBKR for a given symbol and date.

    Results are cached to disk as JSON. Existing cache files from prior Alpaca
    runs are compatible — same format, same filename convention.

    Connects lazily on the first cache miss and reuses the connection across
    calls. Call close() when finished with a backtest run to release the
    connection.
    """

    def __init__(
        self,
        host: str,
        port: int,
        client_id: int,
        cache_dir: str = "backtest_results/cache",
    ) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ib: Optional[IB] = None
        self._last_request: float = 0.0

    def fetch(self, symbol: str, trade_date: date) -> List[Bar]:
        cache_path = self._cache_dir / f"{symbol}_{trade_date}.json"
        if cache_path.exists():
            raw = json.loads(cache_path.read_text())
            return [self._parse_bar(symbol, b) for b in raw]

        bars = self._fetch_from_ibkr(symbol, trade_date)
        if bars is not None:
            cache_path.write_text(json.dumps([self._bar_to_dict(b) for b in bars]))
        return bars or []

    def close(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()
        self._ib = None

    def _ensure_connected(self) -> None:
        if not _HAVE_IBKR:
            raise RuntimeError("ib_insync is required: pip install ib_insync")
        if self._ib is None or not self._ib.isConnected():
            self._ib = IB()
            self._ib.connect(self._host, self._port, clientId=self._client_id)
            logger.info("BarFetcher: connected to IB Gateway")

    def _fetch_from_ibkr(self, symbol: str, trade_date: date) -> Optional[List[Bar]]:
        self._ensure_connected()

        # Pace requests to stay within IBKR's 60-per-10-min limit
        elapsed = time.monotonic() - self._last_request
        if elapsed < _PACING_INTERVAL:
            time.sleep(_PACING_INTERVAL - elapsed)
        self._last_request = time.monotonic()

        contract = Stock(symbol, "SMART", "USD")
        end_dt = datetime(
            trade_date.year, trade_date.month, trade_date.day, 16, 0, tzinfo=_ET
        )

        try:
            ibkr_bars = self._ib.reqHistoricalData(
                contract,
                endDateTime=end_dt,
                durationStr="1 D",
                barSizeSetting="1 min",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=2,       # ib_insync returns datetime objects
                keepUpToDate=False,
            )
        except Exception as exc:
            logger.warning("BarFetcher: IBKR error for %s %s: %s", symbol, trade_date, exc)
            return None

        if not ibkr_bars:
            logger.debug("BarFetcher: no bars returned for %s %s", symbol, trade_date)
            return None

        bars: List[Bar] = []
        for b in ibkr_bars:
            ts = b.date
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            else:
                # Fallback: ib_insync occasionally returns a string
                ts = datetime.fromisoformat(str(ts)).replace(tzinfo=timezone.utc)
            bars.append(Bar(
                symbol=symbol,
                timestamp=ts,
                open=float(b.open),
                high=float(b.high),
                low=float(b.low),
                close=float(b.close),
                volume=int(b.volume),
            ))

        logger.debug("BarFetcher: %d bars fetched for %s %s", len(bars), symbol, trade_date)
        return bars

    def _parse_bar(self, symbol: str, b: dict) -> Bar:
        ts = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
        return Bar(
            symbol=symbol,
            timestamp=ts,
            open=float(b["o"]),
            high=float(b["h"]),
            low=float(b["l"]),
            close=float(b["c"]),
            volume=int(b["v"]),
        )

    def _bar_to_dict(self, bar: Bar) -> dict:
        return {
            "t": bar.timestamp.isoformat(),
            "o": bar.open,
            "h": bar.high,
            "l": bar.low,
            "c": bar.close,
            "v": bar.volume,
        }
