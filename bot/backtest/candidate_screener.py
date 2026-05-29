from __future__ import annotations
import logging
import re
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

from bot.config import V4Config
from bot.data.daily_loader import get_daily_batch

_EXCHANGE_ALLOWLIST = {"NYSE", "NASDAQ", "AMEX"}
_COMMON_STOCK_RE = re.compile(r"^[A-Z]{1,5}$")
_BATCH_SIZE = 500
logger = logging.getLogger(__name__)


class CandidateScreener:
    def __init__(
        self,
        config: V4Config,
        api_key: str,
        secret_key: str,
        base_url: str = "https://paper-api.alpaca.markets",
    ) -> None:
        self._config = config
        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        self._assets_url = f"{base_url.rstrip('/')}/v2/assets"
        self._universe: Optional[List[str]] = None
        # Pre-fetched daily cache keyed by symbol: populated by preload()
        self._daily_cache: Dict[str, pd.DataFrame] = {}

    def preload(self, start: date, end: date) -> None:
        """Download daily bars for the full date range once, used by all subsequent
        candidates_for_date() calls. Avoids per-day repeat downloads."""
        if self._universe is None:
            self._universe = self._load_universe()
        if not self._universe:
            return

        # Extra buffer so we always have a prev-day row for the first date
        fetch_start = (start - timedelta(days=10)).isoformat()
        fetch_end = (end + timedelta(days=1)).isoformat()

        logger.info(
            "CandidateScreener: preloading %d symbols from %s to %s",
            len(self._universe), fetch_start, fetch_end,
        )
        for i in range(0, len(self._universe), _BATCH_SIZE):
            batch = self._universe[i : i + _BATCH_SIZE]
            batch_num = i // _BATCH_SIZE + 1
            total_batches = (len(self._universe) + _BATCH_SIZE - 1) // _BATCH_SIZE
            logger.info("CandidateScreener: preload batch %d/%d", batch_num, total_batches)
            try:
                daily_data = get_daily_batch(batch, start=fetch_start, end=fetch_end)
                self._daily_cache.update(daily_data)
            except Exception as exc:
                logger.warning("CandidateScreener: preload batch %d failed: %s", batch_num, exc)

        logger.info("CandidateScreener: preloaded %d symbols", len(self._daily_cache))

    def candidates_for_date(self, trade_date: date) -> List[str]:
        if self._universe is None:
            self._universe = self._load_universe()
        if not self._universe:
            return []

        target_ts = pd.Timestamp(trade_date)

        # Use pre-loaded cache if available; otherwise fall back to on-demand fetch
        if self._daily_cache:
            daily_data = self._daily_cache
        else:
            start_str = (trade_date - timedelta(days=10)).isoformat()
            end_str = (trade_date + timedelta(days=1)).isoformat()
            daily_data = {}
            for i in range(0, len(self._universe), _BATCH_SIZE):
                batch = self._universe[i : i + _BATCH_SIZE]
                try:
                    daily_data.update(get_daily_batch(batch, start=start_str, end=end_str))
                except Exception as exc:
                    logger.warning("CandidateScreener: batch failed: %s", exc)

        candidates: List[str] = []
        for sym, df in daily_data.items():
            if target_ts not in df.index:
                continue
            idx = df.index.get_loc(target_ts)
            if idx == 0:
                continue
            prev_close = float(df.iloc[idx - 1]["close"])
            day_high = float(df.iloc[idx]["high"])
            day_close = float(df.iloc[idx]["close"])
            if prev_close <= 0:
                continue
            pct_change = (day_high - prev_close) / prev_close
            if (
                pct_change >= self._config.stage1_min_price_change_pct
                and day_close >= self._config.stage1_min_price
            ):
                candidates.append(sym)

        logger.info("CandidateScreener: %d candidates for %s", len(candidates), trade_date)
        return candidates

    def _load_universe(self) -> List[str]:
        try:
            resp = requests.get(
                self._assets_url,
                headers=self._headers,
                params={"status": "active", "asset_class": "us_equity"},
                timeout=30,
            )
            resp.raise_for_status()
            assets = resp.json()
        except Exception as exc:
            logger.error("CandidateScreener: failed to load universe: %s", exc)
            return []

        symbols = []
        for a in assets:
            if not a.get("tradable"):
                continue
            if a.get("exchange") not in _EXCHANGE_ALLOWLIST:
                continue
            sym = a.get("symbol", "")
            if _COMMON_STOCK_RE.match(sym):
                symbols.append(sym)
        logger.info("CandidateScreener: universe loaded with %d symbols", len(symbols))
        return symbols
