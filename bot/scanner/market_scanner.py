from __future__ import annotations
import logging
import time
from datetime import date
from typing import TYPE_CHECKING, List, Optional

import requests

from bot.config import V4Config

if TYPE_CHECKING:
    from bot.scanner.watchlist import Watchlist

logger = logging.getLogger(__name__)

_ASSETS_URL = "https://paper-api.alpaca.markets/v2/assets"
_SNAPSHOTS_URL = "https://data.alpaca.markets/v2/stocks/snapshots"
_BATCH_SIZE = 1000
_EXCHANGE_ALLOWLIST = {"NYSE", "NASDAQ", "AMEX", "ARCA"}


class MarketScanner:
    """Scans the full Alpaca universe for intraday movers using real-time IEX snapshots.

    Universe (all exchange-listed US equities) is loaded once per session from the
    Alpaca assets endpoint and refreshed daily. Snapshots are fetched in batches of
    1000 symbols every scanner_interval_seconds and filtered by Stage 1 criteria.
    """

    def __init__(self, api_key: str, secret_key: str, config: V4Config, watchlist: Watchlist) -> None:
        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        self._cfg = config
        self._watchlist = watchlist
        self._universe: List[str] = []
        self._universe_date: Optional[date] = None

    def scan_once(self) -> None:
        today = date.today()
        if self._universe_date != today:
            self._universe = self._load_universe()
            self._universe_date = today

        for entry in self._fetch_snapshots():
            if entry["percent_change"] < self._cfg.stage1_min_price_change_pct * 100:
                continue
            if entry["price"] < self._cfg.stage1_min_price:
                continue
            self._watchlist.add(entry["symbol"])
            logger.debug("Candidate: %s (%.1f%%)", entry["symbol"], entry["percent_change"])

    def run(self) -> None:
        logger.info("MarketScanner started (interval=%ds)", self._cfg.scanner_interval_seconds)
        while True:
            try:
                self.scan_once()
            except Exception as exc:
                logger.warning("Scanner error: %s", exc)
            time.sleep(self._cfg.scanner_interval_seconds)

    def _load_universe(self) -> List[str]:
        resp = requests.get(
            _ASSETS_URL,
            headers=self._headers,
            params={"status": "active", "asset_class": "us_equity", "tradable": "true"},
            timeout=30,
        )
        resp.raise_for_status()
        symbols = [
            a["symbol"] for a in resp.json()
            if a.get("exchange") in _EXCHANGE_ALLOWLIST
        ]
        logger.info("Universe loaded: %d symbols", len(symbols))
        return symbols

    def _fetch_snapshots(self) -> List[dict]:
        results = []
        for i in range(0, len(self._universe), _BATCH_SIZE):
            batch = self._universe[i:i + _BATCH_SIZE]
            try:
                resp = requests.get(
                    _SNAPSHOTS_URL,
                    headers=self._headers,
                    params={"symbols": ",".join(batch), "feed": "iex"},
                    timeout=15,
                )
                resp.raise_for_status()
                for symbol, snap in resp.json().items():
                    daily = snap.get("dailyBar") or {}
                    prev = snap.get("prevDailyBar") or {}
                    price = daily.get("c", 0.0)
                    prev_close = prev.get("c", 0.0)
                    if prev_close > 0 and price > 0:
                        results.append({
                            "symbol": symbol,
                            "percent_change": (price - prev_close) / prev_close * 100,
                            "price": price,
                            "volume": daily.get("v", 0),
                        })
            except Exception as exc:
                logger.warning("Snapshot batch failed (offset=%d): %s", i, exc)
        return results
