from __future__ import annotations

import asyncio
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

_SNAPSHOT_WAIT_SEC = 3
_SNAPSHOT_RETRY_WAIT_SEC = 5
# Stay well under the typical 100-line concurrent market data limit.
_CHUNK_SIZE = 50


def get_movers_ibkr(host: str, port: int, client_id: int, top_n: int = 200) -> List[dict]:
    """Top gainers via IBKR scanner + reqMktData snapshots.

    Creates and closes its own IB connection — safe to call from any thread.
    Returns same format as get_movers_alpaca:
    [{"symbol": str, "percent_change": float, "price": float, "change": float}]
    """
    if not _HAVE_IBKR:
        raise RuntimeError("ib_insync is required: pip install ib_insync")

    # ib_insync uses asyncio internally; non-main threads have no event loop by default
    # in Python 3.10+, so we must create one before connecting.
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

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

        ib.reqMarketDataType(1)

        mkt_errors: dict[int, tuple[int, str]] = {}

        def _on_error(reqId, errorCode, errorString, contract, advancedOrderRejectJson=""):
            if reqId > 0:
                mkt_errors[reqId] = (errorCode, errorString)
                logger.warning(
                    "Market data error reqId=%d code=%d: %s", reqId, errorCode, errorString
                )

        ib.errorEvent += _on_error

        contracts = [item.contractDetails.contract for item in scan_data]
        all_tickers = []

        # Request in chunks to avoid exceeding concurrent market data line limits.
        for i in range(0, len(contracts), _CHUNK_SIZE):
            chunk = contracts[i : i + _CHUNK_SIZE]
            chunk_tickers = [
                ib.reqMktData(c, genericTickList="", snapshot=True, regulatorySnapshot=False)
                for c in chunk
            ]
            ib.sleep(_SNAPSHOT_WAIT_SEC)
            all_tickers.extend(chunk_tickers)

        # Retry any symbols that came back NaN (slow delivery or transient line-limit hit).
        retry_indices = [
            i for i, t in enumerate(all_tickers)
            if math.isnan(t.last) or math.isnan(t.close)
        ]
        if retry_indices:
            logger.warning("Retrying %d symbols with missing market data", len(retry_indices))
            retry_contracts = [contracts[i] for i in retry_indices]
            retry_tickers = [
                ib.reqMktData(c, genericTickList="", snapshot=True, regulatorySnapshot=False)
                for c in retry_contracts
            ]
            ib.sleep(_SNAPSHOT_RETRY_WAIT_SEC)
            for idx, rt in zip(retry_indices, retry_tickers):
                all_tickers[idx] = rt

        results = []
        for item, ticker in zip(scan_data, all_tickers):
            symbol = item.contractDetails.contract.symbol
            last = ticker.last
            prev_close = ticker.close

            if math.isnan(last) or math.isnan(prev_close) or prev_close == 0:
                logger.warning("No market data for %s after retry — skipping", symbol)
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
