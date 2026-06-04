# broker_alpaca.py
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import requests as _req
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetAssetsRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopOrderRequest,
)
from alpaca.trading.enums import AssetClass, OrderSide, TimeInForce

logger = logging.getLogger(__name__)

# Search from this file upward so the .env is found regardless of cwd
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

if not API_KEY or not API_SECRET:
    raise RuntimeError(
        "Alpaca credentials missing. "
        "Ensure APCA_API_KEY_ID and APCA_API_SECRET_KEY are set in "
        f"{Path(__file__).resolve().parent.parent / '.env'}"
    )

_is_paper = "paper-api" in BASE_URL
trading_client = TradingClient(API_KEY, API_SECRET, paper=_is_paper)

# Rate-limit guard for get_movers (Alpaca enforces ~1 req/15s on this endpoint)
_movers_lock = threading.Lock()
_last_movers_call: float = 0.0
_MOVERS_MIN_INTERVAL = 15.0


# ------------------------------------------------------------------
# Account
# ------------------------------------------------------------------

def get_account_info() -> dict:
    account = trading_client.get_account()
    return {
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "portfolio_value": float(account.portfolio_value),
        "status": account.status,
    }


def get_open_positions() -> dict:
    positions = trading_client.get_all_positions()
    return {p.symbol: float(p.qty) for p in positions}


def get_all_positions_detail() -> dict:
    """Return open positions with full detail needed for reconciliation.

    Returns:
        {symbol: {"qty": int, "entry_price": float, "side": "long"|"short"}}
    """
    positions = trading_client.get_all_positions()
    result = {}
    for p in positions:
        result[p.symbol] = {
            "qty": abs(int(float(p.qty))),
            "entry_price": float(p.avg_entry_price),
            "side": p.side.value if hasattr(p.side, "value") else str(p.side),
        }
    return result


# ------------------------------------------------------------------
# Orders
# ------------------------------------------------------------------

def submit_market_order(symbol: str, qty: float, side: str = "buy") -> str:
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    result = trading_client.submit_order(order)
    return result.id


def submit_limit_order(symbol: str, qty: float, side: str, limit_price: float) -> str:
    order = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        limit_price=round(limit_price, 2),
    )
    result = trading_client.submit_order(order)
    return result.id


def submit_stop_order(symbol: str, qty: float, stop_price: float) -> str:
    """Stop sell order — protects a long position."""
    order = StopOrderRequest(
        symbol=symbol,
        qty=int(qty),  # Alpaca stop orders require whole shares
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        stop_price=round(stop_price, 2),
    )
    result = trading_client.submit_order(order)
    return result.id


def submit_stop_buy_order(symbol: str, qty: float, stop_price: float) -> str:
    """Stop buy order — protects a short position (buy-to-cover if price rises to stop)."""
    order = StopOrderRequest(
        symbol=symbol,
        qty=int(qty),  # Alpaca stop orders require whole shares
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        stop_price=round(stop_price, 2),
    )
    result = trading_client.submit_order(order)
    return result.id


def short_sell(symbol: str, qty: float) -> str:
    """Market sell to open a short position."""
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    result = trading_client.submit_order(order)
    return result.id


def buy_to_cover(symbol: str, qty: float) -> str:
    """Market buy to close a short position."""
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    result = trading_client.submit_order(order)
    return result.id


def buy(symbol: str, qty: float) -> str:
    """Market buy to open a long position."""
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    result = trading_client.submit_order(order)
    return result.id


def sell(symbol: str, qty: float) -> str:
    """Market sell to close a long position."""
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    result = trading_client.submit_order(order)
    return result.id


def cancel_order(order_id: str) -> None:
    trading_client.cancel_order_by_id(order_id)


def get_order(order_id: str):
    return trading_client.get_order_by_id(order_id)


def is_order_filled(order_id: str) -> bool:
    try:
        order = trading_client.get_order_by_id(order_id)
        return "filled" in str(order.status).lower()
    except Exception:
        return False


def get_fill_price(order_id: str, max_wait_sec: float = 2.0) -> Optional[float]:
    """Poll until a market order fills and return the filled average price.

    Returns None if the order hasn't filled within max_wait_sec.
    Market orders on liquid stocks typically fill in milliseconds.
    """
    deadline = time.monotonic() + max_wait_sec
    while time.monotonic() < deadline:
        try:
            order = trading_client.get_order_by_id(order_id)
            if "filled" in str(order.status).lower() and order.filled_avg_price is not None:
                return float(order.filled_avg_price)
        except Exception:
            pass
        time.sleep(0.1)
    return None


# ------------------------------------------------------------------
# Universe / screener
# ------------------------------------------------------------------

def get_etb_set() -> set:
    """Return the set of symbols currently easy-to-borrow for short selling."""
    try:
        assets = trading_client.get_all_assets(
            GetAssetsRequest(asset_class=AssetClass.US_EQUITY, easy_to_borrow=True)
        )
        etb = {a.symbol for a in assets if a.tradable and a.shortable}
        logger.info("ETB list loaded: %d symbols", len(etb))
        return etb
    except Exception as exc:
        logger.error("Failed to fetch ETB list: %s", exc)
        return set()


def get_movers_alpaca(top_n: int = 200) -> list:
    """Return top gainers from Alpaca's screener endpoint.

    Thread-safe rate-limited to _MOVERS_MIN_INTERVAL seconds between calls.
    Each item: {"symbol": str, "percent_change": float, "price": float, "change": float}
    """
    global _last_movers_call

    with _movers_lock:
        elapsed = time.monotonic() - _last_movers_call
        if elapsed < _MOVERS_MIN_INTERVAL:
            time.sleep(_MOVERS_MIN_INTERVAL - elapsed)
        _last_movers_call = time.monotonic()

    resp = _req.get(
        "https://data.alpaca.markets/v1beta1/screener/stocks/movers",
        headers={"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": API_SECRET},
        params={"top": min(top_n, 50)},  # endpoint max is 50
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("gainers", [])
