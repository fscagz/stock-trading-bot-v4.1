from __future__ import annotations
import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
_CACHE_DIR = Path("backtest_results/news_cache")

_CATALYST_KEYWORDS = [
    "fda", "approv", "pdufa", "clearance", "breakthrough",
    "earnings", "eps", "revenue", "beat", "profit", "loss",
    "upgrade", "downgrade", "price target", "outperform", "overweight",
    "acqui", "merger", "buyout", "takeover",
    "contract", "award", "partner", "collaboration", "agreement",
    "phase 1", "phase 2", "phase 3", "clinical", "trial result",
    "offering", "dilut",
    "guidance", "raised guidance", "lowered guidance",
]


class NewsFilter:
    """
    Filters stocks by news catalyst presence on a given date.

    cache_only=True (default): only use the disk cache; cache misses return False
    (no catalyst assumed). This reproduces results from prior runs without new API calls.

    cache_only=False: fetch from Alpaca API on cache miss and persist the result.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        cache_dir: Path = _CACHE_DIR,
        cache_only: bool = True,
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_only = cache_only
        self._mem: dict = {}  # in-process cache to avoid repeated disk reads

    def has_catalyst(self, symbol: str, trade_date: date) -> bool:
        key = (symbol, trade_date)
        if key in self._mem:
            return self._mem[key]
        articles = self._fetch(symbol, trade_date)
        result = self._classify(articles)
        self._mem[key] = result
        return result

    def _fetch(self, symbol: str, trade_date: date) -> list:
        cache_path = self._cache_dir / f"{symbol}_{trade_date}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except Exception:
                pass

        if self._cache_only:
            return []  # treat cache miss as no catalyst — no API call

        start = (trade_date - timedelta(days=1)).isoformat() + "T21:00:00Z"
        end = trade_date.isoformat() + "T21:00:00Z"

        for attempt in range(4):
            try:
                resp = requests.get(
                    _NEWS_URL,
                    headers={
                        "APCA-API-KEY-ID": self._api_key,
                        "APCA-API-SECRET-KEY": self._secret_key,
                    },
                    params={"symbols": symbol, "start": start, "end": end, "limit": 50},
                    timeout=15,
                )
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                articles = resp.json().get("news", [])
                cache_path.write_text(json.dumps(articles))
                return articles
            except Exception as exc:
                if attempt == 3:
                    logger.debug("News fetch failed for %s %s: %s", symbol, trade_date, exc)
                    return []
                time.sleep(2 ** attempt)
        return []

    def _classify(self, articles: list) -> bool:
        for article in articles:
            text = (
                (article.get("headline") or "") + " " +
                (article.get("summary") or "")
            ).lower()
            for kw in _CATALYST_KEYWORDS:
                if kw in text:
                    return True
        return False
