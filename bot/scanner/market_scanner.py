from __future__ import annotations
import logging
import time
from typing import TYPE_CHECKING

import requests

from bot.config import V4Config

if TYPE_CHECKING:
    from bot.scanner.watchlist import Watchlist

logger = logging.getLogger(__name__)

_MOVERS_URL = "https://data.alpaca.markets/v1beta1/screener/stocks/movers"
_ACTIVES_URL = "https://data.alpaca.markets/v1beta1/screener/stocks/most-actives"


class MarketScanner:
    """Polls Alpaca movers + most-actives every N seconds and feeds candidates to the watchlist."""

    def __init__(self, api_key: str, secret_key: str, config: V4Config, watchlist: Watchlist) -> None:
        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        self._cfg = config
        self._watchlist = watchlist

    def scan_once(self) -> None:
        movers = self._fetch_movers()
        actives = self._fetch_most_actives()
        active_symbols = {s["symbol"] for s in actives}

        for entry in movers:
            symbol = entry["symbol"]
            pct_change = entry.get("percent_change", 0.0)
            price = entry.get("price", 0.0)

            if pct_change < self._cfg.stage1_min_price_change_pct * 100:
                continue
            if price < self._cfg.stage1_min_price:
                continue

            high_priority = symbol in active_symbols
            self._watchlist.add(symbol, high_priority=high_priority)
            logger.debug("Candidate: %s (%.1f%%, high_priority=%s)", symbol, pct_change, high_priority)

    def run(self) -> None:
        logger.info("MarketScanner started (interval=%ds)", self._cfg.scanner_interval_seconds)
        while True:
            try:
                self.scan_once()
            except Exception as exc:
                logger.warning("Scanner error: %s", exc)
            time.sleep(self._cfg.scanner_interval_seconds)

    def _fetch_movers(self) -> list:
        resp = requests.get(
            _MOVERS_URL,
            headers=self._headers,
            params={"top": self._cfg.scanner_top_n},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("gainers", [])

    def _fetch_most_actives(self) -> list:
        resp = requests.get(
            _ACTIVES_URL,
            headers=self._headers,
            params={"top": self._cfg.scanner_top_n, "by": "volume"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("most_actives", [])
