"""
Pre-cache 2023 data for backtesting.

Step 1: preload the CandidateScreener for 2023 → saves screener_cache PKL.
Step 2: for each trading day, fetch and cache 1-min bars for all candidates.

Run once; subsequent runs skip already-cached files (BarFetcher checks disk first).
"""
from __future__ import annotations
import copy, os, warnings, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import List

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from bot.backtest.bar_fetcher import BarFetcher
from bot.backtest.candidate_screener import CandidateScreener
from bot.config import make_gap_hold_config

START = date(2023, 1,  3)
END   = date(2023, 12, 29)
WORKERS = 20   # parallel bar-fetch threads

def trading_days(start: date, end: date) -> List[date]:
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days

def main() -> None:
    api_key    = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    base_url   = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    # ── Step 1: screener PKL ─────────────────────────────────────────────────
    print(f"Step 1: loading CandidateScreener for {START} → {END} ...")
    cfg      = make_gap_hold_config()
    screener = CandidateScreener(copy.copy(cfg), api_key, secret_key, base_url)
    screener.preload(START, END)
    print(f"  Screener ready: {len(screener._daily_cache):,} symbols.\n")

    # ── Step 2: bar files ────────────────────────────────────────────────────
    fetcher   = BarFetcher(api_key, secret_key)
    cache_dir = fetcher._cache_dir
    days      = trading_days(START, END)

    # Collect (symbol, date) pairs that aren't cached yet
    to_fetch: List[tuple] = []
    for d in days:
        for sym in screener.candidates_for_date(d):
            if not (cache_dir / f"{sym}_{d}.json").exists():
                to_fetch.append((sym, d))

    total = len(to_fetch)
    already = sum(
        1 for d in days for sym in screener.candidates_for_date(d)
        if (cache_dir / f"{sym}_{d}.json").exists()
    )
    print(f"Step 2: bar files to download: {total:,}  (already cached: {already:,})")
    if total == 0:
        print("  Nothing to do.")
        return

    done = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetcher.fetch, sym, d): (sym, d) for sym, d in to_fetch}
        for fut in as_completed(futures):
            sym, d = futures[fut]
            try:
                bars = fut.result()
                if not bars:
                    errors += 1
            except Exception as exc:
                errors += 1
                print(f"  ERROR {sym} {d}: {exc}")
            done += 1
            if done % 500 == 0 or done == total:
                pct = done / total * 100
                print(f"  {done:,}/{total:,} ({pct:.0f}%)  errors={errors}", flush=True)

    print(f"\nDone. Downloaded {done - errors:,} files, {errors:,} errors.")

if __name__ == "__main__":
    main()
