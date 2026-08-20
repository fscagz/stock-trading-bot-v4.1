"""
Build a 2017-06 → 2025-01 daily-bars pkl (screener_cache format) via Alpaca SIP.

Purpose: multi-regime validation of the institutional-PEAD long (two bears:
2018-Q4, 2022; one crash: 2020-Q2; two bulls) — the 2024-26 pkl alone cannot
distinguish edge from bull beta. Daily bars only; no intraday fetching.

Universe: current active Alpaca assets (survivorship bias — both delisted
losers AND acquired winners are absent; disclosed in the analysis). SPY is
appended for benchmark-adjusted returns.
"""
from __future__ import annotations
import logging, os, pickle, re, sys, time
from datetime import date
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_EXCHANGE_ALLOWLIST = {"NYSE", "NASDAQ", "AMEX"}
_COMMON_STOCK_RE = re.compile(r"^[A-Z]{1,5}$")
_ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
_SCREENER_CACHE_DIR = Path("screener_cache")
_BATCH_SIZE = 100

START = date(2017, 6, 1)    # 100-day warmup before 2018 events
END = date(2025, 1, 31)


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
    raw_bars: Dict[str, list] = {}
    params: dict = {
        "symbols": ",".join(symbols),
        "timeframe": "1Day",
        "start": start,
        "end": end,
        "limit": 10000,
        "feed": "sip",
        "adjustment": "split",   # split-adjusted so gaps aren't fake split artifacts
    }
    while True:
        for attempt in range(5):
            try:
                resp = requests.get(_ALPACA_BARS_URL, headers=get_headers(), params=params, timeout=60)
                if resp.status_code == 429:
                    wait = min(120, 10 * 2 ** attempt)
                    logger.warning("rate limited, retrying in %ds", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                if attempt == 4:
                    logger.error("gave up on batch: %s", exc)
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


def main():
    pkl_path = _SCREENER_CACHE_DIR / f"{START}_{END}.pkl"
    if pkl_path.exists():
        logger.info("%s already exists — delete it to rebuild", pkl_path)
        return

    universe = load_universe()
    universe.append("SPY")  # benchmark
    daily_cache: Dict[str, pd.DataFrame] = {}
    total_batches = (len(universe) + _BATCH_SIZE - 1) // _BATCH_SIZE
    for i in range(0, len(universe), _BATCH_SIZE):
        batch = universe[i:i + _BATCH_SIZE]
        batch_num = i // _BATCH_SIZE + 1
        if batch_num % 10 == 0 or batch_num == 1:
            logger.info("batch %d/%d (%d symbols cached)", batch_num, total_batches, len(daily_cache))
        daily_cache.update(fetch_daily_sip(batch, START.isoformat(), END.isoformat()))

    _SCREENER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as f:
        pickle.dump(daily_cache, f)
    logger.info("saved %d symbols → %s", len(daily_cache), pkl_path)


if __name__ == "__main__":
    main()
