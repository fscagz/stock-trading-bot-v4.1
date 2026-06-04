from __future__ import annotations

import logging
import math
import time
from typing import List

logger = logging.getLogger(__name__)

try:
    from ib_insync import IB, ScannerSubscription
    _HAVE_IBKR = True
except ImportError:
    _HAVE_IBKR = False

# How long to wait for snapshot tick data after reqMktData
_SNAPSHOT_WAIT_SEC = 2


def get_movers_ibkr(host: str, port: int, client_id: int, top_n: int = 200) -> List[dict]:
    """Top gainers via IBKR scanner + reqMktData snapshots.

    Creates and closes its own IB connection — safe to call from any thread.
    Returns same format as get_movers_alpaca:
    [{"symbol": str, "percent_change": float, "price": float, "change": float}]
    """
    if not _HAVE_IBKR:
        raise RuntimeError("ib_insync is required: pip install ib_insync")

    ib = IB()
    ib.connect(host, port, clientId=client_id)
    try:
        # abovePrice filters penny stocks server-side so we don't waste subscription slots.
        sub = ScannerSubscription(
            instrument="STK",
            locationCode="STK.US.MAJOR",
            scanCode="TOP_PERC_GAIN",
            numberOfRows=top_n,
            abovePrice=1.0,
        )
        scan_data = ib.reqScannerData(sub)
        if not scan_data:
            logger.debug("IBKR scanner: no results (market may be closed)")
            return []

        logger.debug("IBKR scanner: %d symbols returned", len(scan_data))

        # Use delayed market data (type 3) — real-time requires a separate streaming
        # subscription beyond our historical-data plan, but delayed gives us the fields
        # we need: ticker.last (current price) and ticker.close (previous day's close).
        ib.reqMarketDataType(3)

        # Batch-request snapshots for all symbols, then wait once
        contracts = [item.contractDetails.contract for item in scan_data]
        tickers = [
            ib.reqMktData(c, genericTickList="", snapshot=True, regulatorySnapshot=False)
            for c in contracts
        ]
        ib.sleep(_SNAPSHOT_WAIT_SEC)

        results = []
        for item, ticker in zip(scan_data, tickers):
            symbol = item.contractDetails.contract.symbol
            last = ticker.last
            prev_close = ticker.close

            if math.isnan(last) or math.isnan(prev_close) or prev_close == 0:
                continue

            change = last - prev_close
            pct = change / prev_close * 100
            results.append({
                "symbol": symbol,
                "percent_change": round(pct, 4),
                "price": round(last, 4),
                "change": round(change, 4),
            })

        return results
    finally:
        ib.disconnect()


def get_movers(host: str, port: int, client_id: int, top_n: int = 200) -> List[dict]:
    """Top gainers: IBKR scanner primary, Alpaca screener fallback.

    Returns same format as get_movers_alpaca.
    """
    try:
        results = get_movers_ibkr(host, port, client_id, top_n)
        logger.info("IBKR scanner: %d movers", len(results))
        return results
    except Exception as exc:
        logger.warning("IBKR scanner failed: %s — falling back to Alpaca screener", exc)

    import bot.broker_alpaca as _alpaca
    return _alpaca.get_movers_alpaca(top_n)
