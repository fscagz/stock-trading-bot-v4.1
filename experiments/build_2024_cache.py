"""
Build 2024 bar cache using Alpaca SIP feed.

Steps:
  1. Delete existing 2024 screener preload (built with yfinance/iex, unreliable).
  2. Rebuild screener preload for 2023-11-27..2024-12-31 using SIP feed.
  3. For each 2024 trading day, find candidates (15%+ intraday, $250k avg vol).
  4. Fetch 1-min bars from Alpaca SIP for any uncached candidate/date pair.
"""
from __future__ import annotations
import json
import logging
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
_EXCHANGE_ALLOWLIST = {"NYSE", "NASDAQ", "AMEX"}
_COMMON_STOCK_RE = re.compile(r"^[A-Z]{1,5}$")
_ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
_ALPACA_SINGLE_BARS_URL = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
_SCREENER_CACHE_DIR = Path("screener_cache")
_BAR_CACHE_DIR = Path("backtest_results/cache")
_BATCH_SIZE = 100

START = date(2024, 1, 2)
END = date(2024, 12, 31)
PRELOAD_START = date(2023, 11, 27)  # 35-day buffer before 2024
PRELOAD_END = date(2025, 1, 1)

# Long strategy thresholds (matches make_gap_hold_config)
STAGE1_MIN_PCT_CHANGE = 0.15
STAGE1_MIN_PRICE = 2.00
MIN_AVG_DOLLAR_VOLUME = 250_000


def trading_days(start: date, end: date) -> List[date]:
    days, current = [], start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def get_headers() -> dict:
    return {
        "APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"],
        "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"],
    }


def load_universe() -> List[str]:
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    resp = requests.get(
        f"{base_url}/v2/assets",
        headers=get_headers(),
        params={"status": "active", "asset_class": "us_equity"},
        timeout=30,
    )
    resp.raise_for_status()
    symbols = []
    for a in resp.json():
        if not a.get("tradable"):
            continue
        if a.get("exchange") not in _EXCHANGE_ALLOWLIST:
            continue
        sym = a.get("symbol", "")
        if _COMMON_STOCK_RE.match(sym):
            symbols.append(sym)
    logger.info("Universe: %d symbols", len(symbols))
    return symbols


def fetch_daily_sip(symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Fetch daily OHLCV bars via Alpaca SIP feed for a batch of symbols."""
    raw_bars: Dict[str, list] = {}
    params: dict = {
        "symbols": ",".join(symbols),
        "timeframe": "1Day",
        "start": start,
        "end": end,
        "limit": 10000,
        "feed": "sip",
    }
    while True:
        for attempt in range(5):
            try:
                resp = requests.get(
                    _ALPACA_BARS_URL, headers=get_headers(), params=params, timeout=60,
                )
                if resp.status_code == 429:
                    wait = min(120, 10 * 2 ** attempt)
                    logger.warning("Daily SIP: rate limited, retrying in %ds", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                if attempt == 4:
                    logger.error("Daily SIP: gave up on batch: %s", exc)
                    return {}
                time.sleep(5 * 2 ** attempt)

        for sym, bars in (data.get("bars") or {}).items():
            raw_bars.setdefault(sym, []).extend(bars)

        next_token = data.get("next_page_token")
        if not next_token:
            break
        params["page_token"] = next_token

    result: Dict[str, pd.DataFrame] = {}
    for sym, bars in raw_bars.items():
        if not bars:
            continue
        df = pd.DataFrame(bars)
        df["t"] = pd.to_datetime(df["t"]).dt.tz_localize(None).dt.normalize()
        df = df.rename(columns={"t": "date", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
        result[sym] = df
    return result


def build_screener_preload(universe: List[str]) -> Dict[str, pd.DataFrame]:
    """Rebuild the 2022 screener preload using SIP daily bars."""
    preload_key = f"{PRELOAD_START}_{PRELOAD_END}"
    pkl_path = _SCREENER_CACHE_DIR / f"{preload_key}.pkl"
    _SCREENER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Always rebuild to ensure SIP data (replaces any yfinance-based PKL)
    if pkl_path.exists():
        logger.info("Removing existing screener preload to rebuild with SIP feed...")
        pkl_path.unlink()

    logger.info("Building screener preload for %s..%s using SIP feed (%d symbols)",
                PRELOAD_START, PRELOAD_END, len(universe))
    daily_cache: Dict[str, pd.DataFrame] = {}
    total_batches = (len(universe) + _BATCH_SIZE - 1) // _BATCH_SIZE

    for i in range(0, len(universe), _BATCH_SIZE):
        batch = universe[i:i + _BATCH_SIZE]
        batch_num = i // _BATCH_SIZE + 1
        if batch_num % 20 == 0 or batch_num == 1:
            logger.info("Preload batch %d/%d (%d symbols so far)", batch_num, total_batches, len(daily_cache))
        data = fetch_daily_sip(batch, PRELOAD_START.isoformat(), PRELOAD_END.isoformat())
        daily_cache.update(data)

    with pkl_path.open("wb") as f:
        pickle.dump(daily_cache, f)
    logger.info("Screener preload saved: %d symbols → %s", len(daily_cache), pkl_path)
    return daily_cache


def candidates_for_date(trade_date: date, daily_cache: Dict[str, pd.DataFrame]) -> List[str]:
    """Find symbols with 15%+ intraday move and $250k avg dollar volume on trade_date."""
    target_ts = pd.Timestamp(trade_date)
    scored = []
    for sym, df in daily_cache.items():
        if target_ts not in df.index:
            continue
        idx = df.index.get_loc(target_ts)
        if idx == 0:
            continue
        prev_close = float(df.iloc[idx - 1]["close"])
        day_high = float(df.iloc[idx]["high"])
        day_close = float(df.iloc[idx]["close"])
        if prev_close <= 0 or day_close < STAGE1_MIN_PRICE:
            continue
        pct_change = (day_high - prev_close) / prev_close
        past = df[df.index < target_ts]
        avg_dollar_vol = float(past["volume"].tail(20).mean() * day_close) if not past.empty else 0.0
        if pct_change >= STAGE1_MIN_PCT_CHANGE and avg_dollar_vol >= MIN_AVG_DOLLAR_VOLUME:
            scored.append((pct_change, sym))
    scored.sort(reverse=True)
    return [sym for _, sym in scored[:200]]


def fetch_intraday_sip(symbol: str, trade_date: date) -> Optional[list]:
    """Fetch 1-min bars for symbol on trade_date using Alpaca SIP feed."""
    start = datetime(trade_date.year, trade_date.month, trade_date.day, 9, 30, tzinfo=_ET)
    end = datetime(trade_date.year, trade_date.month, trade_date.day, 16, 0, tzinfo=_ET)
    url = _ALPACA_SINGLE_BARS_URL.format(symbol=symbol)
    params: dict = {
        "timeframe": "1Min",
        "feed": "sip",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": 1000,
    }
    all_bars = []
    while True:
        for attempt in range(4):
            try:
                resp = requests.get(url, headers=get_headers(), params=params, timeout=30)
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                if attempt == 3:
                    logger.warning("Intraday SIP: gave up on %s %s: %s", symbol, trade_date, exc)
                    return None
        else:
            return None

        for b in data.get("bars") or []:
            ts = datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(_ET)
            t = (ts.hour, ts.minute)
            if (9, 30) <= t <= (16, 0):
                all_bars.append({
                    "t": b["t"], "o": b["o"], "h": b["h"],
                    "l": b["l"], "c": b["c"], "v": b["v"],
                })

        next_token = data.get("next_page_token")
        if not next_token:
            break
        params["page_token"] = next_token

    return all_bars


def main():
    _BAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Step 1: Load universe ===")
    universe = load_universe()

    logger.info("=== Step 2: Build screener preload (SIP daily bars) ===")
    daily_cache = build_screener_preload(universe)

    logger.info("=== Step 3: Identify 2024 candidates and fetch intraday bars ===")
    days = trading_days(START, END)
    logger.info("Processing %d trading days in 2024", len(days))

    total_fetched = 0
    total_cached = 0
    total_empty = 0

    for d in days:
        cands = candidates_for_date(d, daily_cache)
        if not cands:
            continue

        missing = [sym for sym in cands if not (_BAR_CACHE_DIR / f"{sym}_{d}.json").exists()]
        if not missing:
            total_cached += len(cands)
            continue

        logger.info("%s: %d candidates, %d to fetch", d, len(cands), len(missing))

        def _fetch(sym: str):
            bars = fetch_intraday_sip(sym, d)
            return sym, bars

        # Sequential to respect rate limits — Alpaca allows ~200 req/min on data API
        for sym in missing:
            bars = fetch_intraday_sip(sym, d)
            cache_path = _BAR_CACHE_DIR / f"{sym}_{d}.json"
            if bars is not None:
                cache_path.write_text(json.dumps(bars))
                if bars:
                    total_fetched += 1
                else:
                    total_empty += 1
                    cache_path.unlink()  # don't cache empty files
            time.sleep(0.15)  # ~7 req/sec, well under 200/min limit

    logger.info("Done. Fetched: %d bars files | Already cached: %d | Empty/failed: %d",
                total_fetched, total_cached, total_empty)
    logger.info("2024 cache is ready. Run: python3 -m bot.backtest --long --regime --start 2024-01-02 --end 2024-12-31")


if __name__ == "__main__":
    main()
