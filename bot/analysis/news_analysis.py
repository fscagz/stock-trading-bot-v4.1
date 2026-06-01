"""
Retroactive news catalyst analysis.

For each trade in the backtest CSV, fetches Alpaca news for that ticker
on the trade date, classifies whether a meaningful catalyst was present,
then compares PnL between catalyst vs. no-catalyst trades.

Usage:
    python3.13 -m bot.analysis.news_analysis
"""
from __future__ import annotations
import csv
import json
import logging
import os
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
_CACHE_DIR = Path("backtest_results/news_cache")
_TRADES_CSV = Path("backtest_results/trades_2026-03-01_2026-05-28.csv")

# Keywords that indicate a meaningful fundamental catalyst
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


def _fetch_news(symbol: str, trade_date: date, api_key: str, secret_key: str) -> list:
    cache_path = _CACHE_DIR / f"{symbol}_{trade_date}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    # Window: previous calendar day through end of trade date
    start = (trade_date - timedelta(days=1)).isoformat() + "T21:00:00Z"
    end = trade_date.isoformat() + "T21:00:00Z"

    for attempt in range(4):
        try:
            resp = requests.get(
                _NEWS_URL,
                headers={
                    "APCA-API-KEY-ID": api_key,
                    "APCA-API-SECRET-KEY": secret_key,
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
                logger.warning("News fetch failed for %s %s: %s", symbol, trade_date, exc)
                return []
            time.sleep(2 ** attempt)
    return []


def _classify(articles: list) -> tuple[bool, list[str]]:
    """Return (has_catalyst, matched_keywords)."""
    matched = []
    for article in articles:
        text = (
            (article.get("headline") or "") + " " +
            (article.get("summary") or "")
        ).lower()
        for kw in _CATALYST_KEYWORDS:
            if kw in text and kw not in matched:
                matched.append(kw)
    return bool(matched), matched


def main() -> None:
    api_key = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    trades = []
    with _TRADES_CSV.open() as f:
        for row in csv.DictReader(f):
            row["pnl"] = float(row["pnl"]) if row["pnl"] else 0.0
            row["entry_price"] = float(row["entry_price"])
            entry_dt = datetime.fromisoformat(row["entry_time"])
            row["trade_date"] = entry_dt.date()
            trades.append(row)

    logger.info("Fetching news for %d trades...", len(trades))

    catalyst: list = []
    no_catalyst: list = []
    annotated = []

    seen = {}
    for i, t in enumerate(trades):
        key = (t["ticker"], t["trade_date"])
        if key not in seen:
            articles = _fetch_news(t["ticker"], t["trade_date"], api_key, secret_key)
            has_cat, keywords = _classify(articles)
            seen[key] = (has_cat, keywords, len(articles))
        has_cat, keywords, n_articles = seen[key]
        t["has_catalyst"] = has_cat
        t["catalyst_keywords"] = keywords
        t["n_articles"] = n_articles
        (catalyst if has_cat else no_catalyst).append(t)
        if (i + 1) % 50 == 0:
            logger.info("  %d/%d processed", i + 1, len(trades))

    def _stats(group: list, label: str) -> None:
        if not group:
            print(f"  {label}: no trades")
            return
        wins = sum(1 for t in group if t["pnl"] > 0)
        total_pnl = sum(t["pnl"] for t in group)
        avg = total_pnl / len(group)
        avg_win = sum(t["pnl"] for t in group if t["pnl"] > 0) / max(wins, 1)
        avg_loss = sum(t["pnl"] for t in group if t["pnl"] <= 0) / max(len(group) - wins, 1)
        print(
            f"  {label:20s}  n={len(group):3d}  "
            f"win%={wins/len(group)*100:.0f}%  "
            f"avg={avg:+.2f}  "
            f"total={total_pnl:+.2f}  "
            f"avg_win={avg_win:+.2f}  avg_loss={avg_loss:+.2f}"
        )

    print("\n=== News Catalyst Analysis ===")
    _stats(catalyst, "WITH catalyst")
    _stats(no_catalyst, "WITHOUT catalyst")

    # Most common catalyst keywords
    kw_counts: dict = defaultdict(int)
    for t in catalyst:
        for kw in t["catalyst_keywords"]:
            kw_counts[kw] += 1
    print("\nTop catalyst keywords found:")
    for kw, count in sorted(kw_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {kw:30s}  {count} trades")

    # Exit reason breakdown by catalyst
    print("\nExit reasons — WITH catalyst:")
    by_reason: dict = defaultdict(list)
    for t in catalyst:
        by_reason[t["exit_reason"]].append(t["pnl"])
    for reason, pnls in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        wins = sum(1 for p in pnls if p > 0)
        print(f"  {reason:20s}  n={len(pnls):3d}  win%={wins/len(pnls)*100:.0f}%  total={sum(pnls):+.2f}")

    print("\nExit reasons — WITHOUT catalyst:")
    by_reason2: dict = defaultdict(list)
    for t in no_catalyst:
        by_reason2[t["exit_reason"]].append(t["pnl"])
    for reason, pnls in sorted(by_reason2.items(), key=lambda x: -len(x[1])):
        wins = sum(1 for p in pnls if p > 0)
        print(f"  {reason:20s}  n={len(pnls):3d}  win%={wins/len(pnls)*100:.0f}%  total={sum(pnls):+.2f}")

    # Save annotated CSV
    out_path = Path("backtest_results/trades_news_annotated.csv")
    fieldnames = list(trades[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trades)
    print(f"\nAnnotated trades saved to {out_path}")


if __name__ == "__main__":
    main()
