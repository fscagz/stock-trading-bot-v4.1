from __future__ import annotations
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

import requests

from bot.intraday.types import Bar

_ET = ZoneInfo("America/New_York")
_SESSION_OPEN = (9, 30)
_SESSION_CLOSE = (16, 0)
_BARS_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
logger = logging.getLogger(__name__)


class BarFetcher:
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        cache_dir: str = "backtest_results/cache",
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, symbol: str, trade_date: date) -> List[Bar]:
        cache_path = self._cache_dir / f"{symbol}_{trade_date}.json"
        if cache_path.exists():
            raw = json.loads(cache_path.read_text())
            return [self._parse_bar(symbol, b) for b in raw]

        bars = self._fetch_from_api(symbol, trade_date)
        if bars is not None:
            cache_path.write_text(json.dumps([self._bar_to_dict(b) for b in bars]))
        return bars or []

    def _fetch_from_api(self, symbol: str, trade_date: date) -> Optional[List[Bar]]:
        start = datetime(
            trade_date.year, trade_date.month, trade_date.day, 9, 30, tzinfo=_ET
        )
        end = datetime(
            trade_date.year, trade_date.month, trade_date.day, 16, 0, tzinfo=_ET
        )
        url = _BARS_URL.format(symbol=symbol)
        headers = {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
        }
        params: dict = {
            "timeframe": "1Min",
            "feed": "iex",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 1000,
        }

        all_bars: List[Bar] = []
        while True:
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("BarFetcher: API error for %s %s: %s", symbol, trade_date, exc)
                return None

            for b in data.get("bars") or []:
                bar = self._parse_bar(symbol, b)
                if self._in_session(bar.timestamp):
                    all_bars.append(bar)

            next_token = data.get("next_page_token")
            if not next_token:
                break
            params["page_token"] = next_token

        return all_bars

    def _in_session(self, ts: datetime) -> bool:
        ts_et = ts.astimezone(_ET)
        t = (ts_et.hour, ts_et.minute)
        return _SESSION_OPEN <= t <= _SESSION_CLOSE

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
