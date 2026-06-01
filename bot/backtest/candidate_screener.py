from __future__ import annotations
import logging
import pickle
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from bot.config import V4Config
from bot.data.daily_loader import get_daily_batch

_SCREENER_CACHE_DIR = Path("screener_cache")

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
        # Pre-computed candidates per date: populated by preload()
        self._candidates_index: Dict[date, List[str]] = {}

    def preload(self, start: date, end: date) -> None:
        """Load daily bars for the date range.

        Fast path: use a pre-built SIP pkl from screener_cache/ if one fully covers
        the requested range (built by build_2022_cache.py or equivalent).
        Slow path: download from Yahoo Finance in batches.
        """
        if self._universe is None:
            self._universe = self._load_universe()

        fetch_start = start - timedelta(days=10)
        pkl = self._find_pkl(fetch_start, end)
        if pkl is not None:
            with pkl.open("rb") as f:
                self._daily_cache = pickle.load(f)
            logger.info("CandidateScreener: loaded %d symbols from %s",
                        len(self._daily_cache), pkl.name)
            self._build_candidates_index(start, end)
            return

        if not self._universe:
            return
        fetch_start_str = fetch_start.isoformat()
        fetch_end_str = (end + timedelta(days=1)).isoformat()
        logger.info(
            "CandidateScreener: preloading %d symbols from %s to %s via yfinance",
            len(self._universe), fetch_start_str, fetch_end_str,
        )
        for i in range(0, len(self._universe), _BATCH_SIZE):
            batch = self._universe[i : i + _BATCH_SIZE]
            batch_num = i // _BATCH_SIZE + 1
            total_batches = (len(self._universe) + _BATCH_SIZE - 1) // _BATCH_SIZE
            logger.info("CandidateScreener: preload batch %d/%d", batch_num, total_batches)
            try:
                daily_data = get_daily_batch(batch, start=fetch_start_str, end=fetch_end_str)
                self._daily_cache.update(daily_data)
            except Exception as exc:
                logger.warning("CandidateScreener: preload batch %d failed: %s", batch_num, exc)
        logger.info("CandidateScreener: preloaded %d symbols", len(self._daily_cache))
        # Save PKL so future runs use the same candidates (makes backtests deterministic)
        _SCREENER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pkl_path = _SCREENER_CACHE_DIR / f"{fetch_start}_{end}.pkl"
        try:
            with pkl_path.open("wb") as f:
                pickle.dump(self._daily_cache, f)
            logger.info("CandidateScreener: saved PKL to %s", pkl_path.name)
        except Exception as exc:
            logger.warning("CandidateScreener: failed to save PKL: %s", exc)
        self._build_candidates_index(start, end)

    def _build_candidates_index(self, start: date, end: date) -> None:
        """Pre-compute qualifying symbol lists for every trading day in [start, end].

        Vectorized per-symbol: runs once at preload time so candidates_for_date() is O(1).
        """
        min_pct = self._config.stage1_min_price_change_pct
        min_price = self._config.stage1_min_price
        min_dv = getattr(self._config, "min_avg_dollar_volume", 0.0)

        days: List[date] = []
        d = start
        while d <= end:
            if d.weekday() < 5:
                days.append(d)
            d += timedelta(days=1)
        day_set = set(pd.Timestamp(d) for d in days)

        index: Dict[date, List[str]] = {d: [] for d in days}

        for sym, df in self._daily_cache.items():
            if df.empty or len(df) < 2:
                continue
            prev_close = df["close"].shift(1)
            pct_chg = (df["high"] - prev_close) / prev_close.replace(0, float("nan"))
            passes_price = (pct_chg >= min_pct) & (df["close"] >= min_price)
            if min_dv > 0:
                avg_vol = df["volume"].rolling(20, min_periods=1).mean()
                passes_price = passes_price & (avg_vol * df["close"] >= min_dv)
            qualifying_ts = df.index[passes_price.fillna(False)]
            for ts in qualifying_ts:
                if ts in day_set:
                    index[ts.date()].append(sym)

        self._candidates_index = index
        logger.info("CandidateScreener: index built — %d days, avg %.0f candidates/day",
                    len(index), sum(len(v) for v in index.values()) / max(len(index), 1))

    def _passes_dollar_volume(self, df: pd.DataFrame, target_ts: pd.Timestamp, day_close: float) -> bool:
        """Return True if 20-day avg daily dollar volume meets config.min_avg_dollar_volume."""
        min_dv = getattr(self._config, "min_avg_dollar_volume", 0.0)
        if min_dv <= 0:
            return True
        past = df[df.index < target_ts]
        if past.empty:
            return False
        avg_vol = float(past["volume"].tail(20).mean())
        return avg_vol * day_close >= min_dv

    def _find_pkl(self, start: date, end: date) -> Optional[Path]:
        """Return the smallest pkl file in screener_cache/ that fully covers [start, end]."""
        if not _SCREENER_CACHE_DIR.exists():
            return None
        _date_re = re.compile(r"^(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})$")
        best: Optional[tuple] = None
        for pkl in _SCREENER_CACHE_DIR.glob("*.pkl"):
            m = _date_re.match(pkl.stem)
            if not m:
                continue
            try:
                pkl_start = date.fromisoformat(m.group(1))
                pkl_end = date.fromisoformat(m.group(2))
            except ValueError:
                continue
            if pkl_start <= start and pkl_end >= end:
                span = (pkl_end - pkl_start).days
                if best is None or span < best[0]:
                    best = (span, pkl)
        return best[1] if best else None

    def baseline_volume(self, sym: str, trade_date: date) -> float:
        """20-day per-minute average volume before trade_date (used by run_experiments)."""
        df = self._daily_cache.get(sym)
        if df is None or df.empty:
            return 0.0
        past = df[df.index < pd.Timestamp(trade_date)]
        return float(past["volume"].tail(20).mean()) / 390 if not past.empty else 0.0

    def prior_close(self, sym: str, trade_date: date) -> float:
        """Previous trading day's close (used by run_experiments)."""
        df = self._daily_cache.get(sym)
        if df is None or df.empty:
            return 0.0
        past = df[df.index < pd.Timestamp(trade_date)]
        return float(past.iloc[-1]["close"]) if not past.empty else 0.0

    def candidates_for_date(self, trade_date: date) -> List[str]:
        # Fast path: use precomputed index built during preload()
        if trade_date in self._candidates_index:
            candidates = self._candidates_index[trade_date]
            logger.info("CandidateScreener: %d candidates for %s", len(candidates), trade_date)
            return candidates

        # Slow path: on-demand compute (single-day mode or preload not called)
        if self._universe is None:
            self._universe = self._load_universe()
        if not self._universe:
            return []

        target_ts = pd.Timestamp(trade_date)
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
            if pct_change >= self._config.stage1_min_price_change_pct and day_close >= self._config.stage1_min_price:
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
