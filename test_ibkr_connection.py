"""
Diagnostic script — tests every IBKR API call the bot uses.
Reads IBKR_HOST and IBKR_PORT from .env automatically.
"""
from __future__ import annotations
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from ib_insync import IB, Stock

load_dotenv(Path(__file__).resolve().parent / ".env")

_ET = ZoneInfo("America/New_York")
HOST = os.getenv("IBKR_HOST", "127.0.0.1")
PORT = int(os.getenv("IBKR_PORT", "4001"))
OK = "✓"
FAIL = "✗"


def section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


def last_trading_day() -> date:
    """Most recent weekday on or before today."""
    d = date.today()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def test_historical_bars(ib: IB) -> bool:
    section("Historical 1-min bars (BarFetcher)")
    trade_date = last_trading_day()
    print(f"  Symbol : AAPL")
    print(f"  Date   : {trade_date}")

    contract = Stock("AAPL", "SMART", "USD")
    end_dt = datetime(trade_date.year, trade_date.month, trade_date.day, 16, 0, tzinfo=_ET)

    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_dt,
            durationStr="1 D",
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=2,
            keepUpToDate=False,
        )
    except Exception as exc:
        print(f"  {FAIL} Request failed: {exc}")
        return False

    if not bars:
        print(f"  {FAIL} No bars returned (market may have been closed on {trade_date})")
        return False

    first, last = bars[0], bars[-1]
    print(f"  {OK} {len(bars)} bars returned")
    print(f"  First bar : {first.date}  O={first.open:.2f} H={first.high:.2f} L={first.low:.2f} C={first.close:.2f} V={first.volume}")
    print(f"  Last bar  : {last.date}  O={last.open:.2f} H={last.high:.2f} L={last.low:.2f} C={last.close:.2f} V={last.volume}")

    # Sanity checks
    issues = []
    if len(bars) < 300:
        issues.append(f"only {len(bars)} bars — expected ~390 for a full session")
    if first.volume <= 0:
        issues.append("first bar has zero volume")
    if first.close <= 0:
        issues.append("first bar has zero close price")

    if issues:
        for i in issues:
            print(f"  ⚠ {i}")
    else:
        print(f"  {OK} All sanity checks passed")

    return True


def test_realtime_bars(ib: IB) -> bool:
    section("Real-time 1-min bars via keepUpToDate (BarStream)")
    print("  Subscribing to AAPL for 15 seconds using reqHistoricalData keepUpToDate=True ...")

    received = []
    contract = Stock("AAPL", "SMART", "USD")

    bar_list = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="1 D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=2,
        keepUpToDate=True,
    )

    def on_update(bars, has_new_bar: bool) -> None:
        if has_new_bar and len(bars) >= 2:
            b = bars[-2]  # last *completed* bar
            received.append(b)
            print(f"  {OK} Bar: time={b.date}  O={b.open:.2f} H={b.high:.2f} L={b.low:.2f} C={b.close:.2f} V={b.volume}")

    bar_list.updateEvent += on_update

    try:
        ib.sleep(15)
    finally:
        bar_list.updateEvent -= on_update
        ib.cancelHistoricalData(bar_list)

    now_et = datetime.now(_ET)
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    is_market_hours = now_et.weekday() < 5 and market_open <= now_et <= market_close

    if not received:
        if not is_market_hours:
            print(f"  ⚠ No bars received — market is closed right now (expected outside trading hours)")
            print(f"  {OK} Subscription connected without errors — bars will arrive during market hours")
            return True
        else:
            print(f"  {FAIL} No bars received during market hours")
            return False

    print(f"  {OK} {len(received)} completed bar(s) received in 15 seconds")
    return True


def main() -> None:
    print("\n╔══════════════════════════════════════════════╗")
    print("║     IBKR API Diagnostic — stock-trading-bot  ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"\nConnecting to IB Gateway at {HOST}:{PORT} ...")

    ib = IB()
    try:
        ib.connect(HOST, PORT, clientId=99)
    except Exception as exc:
        print(f"\n{FAIL} Could not connect to IB Gateway: {exc}")
        print("  Make sure IB Gateway is open, logged in, and API is enabled on port 4002.")
        sys.exit(1)

    print(f"{OK} Connected — account: {ib.wrapper.accounts}")

    results = {}
    try:
        results["historical"] = test_historical_bars(ib)
        results["realtime"]   = test_realtime_bars(ib)
    finally:
        ib.disconnect()
        print(f"\n{'─' * 50}")

    passed = sum(results.values())
    total  = len(results)
    print(f"\n  Results: {passed}/{total} tests passed")
    for name, ok in results.items():
        print(f"    {'✓' if ok else '✗'}  {name}")
    print()
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
