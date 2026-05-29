# broker_alpaca.py
'''
 Initializes Alpaca TradingClient using API credentials.
 Input: None (reads from environment variables or hardcoded BASE_URL)
 Output: trading_client object for submitting orders and retrieving account data
'''

import os
# from dotenv import load_dotenv # Might not be needed depending on Python installation
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# load_dotenv()
'''
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL")
'''
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

def get_account_info():
    '''
 Retrieves account information such as cash, buying power, portfolio value, and account status.
 Input: None
 Output: Dictionary with account metrics (cash, buying_power, portfolio_value, status)
    '''
    account = trading_client.get_account()
    return {
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "portfolio_value": float(account.portfolio_value),
        "status": account.status
    }

def get_open_positions():
    '''
 Fetches all currently open positions in the Alpaca account.
 Input: None
 Output: Dictionary mapping ticker symbols to position quantities (as floats)
    '''
    positions = trading_client.get_all_positions()
    return {p.symbol: float(p.qty) for p in positions}

def submit_market_order(symbol, qty, side = "buy"):
    '''
 Submits a market order to buy or sell a specified quantity of a stock.
 Input:
   - symbol (str): Ticker symbol of the stock
   - qty (int or float): Quantity of shares to trade
   - side (str): "buy" or "sell" (default is "buy")
 Output:
   - str: Order ID of the submitted market order
    '''
    order = MarketOrderRequest(
        symbol = symbol,
        qty = qty,
        side = OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force = TimeInForce.DAY
    )
    result = trading_client.submit_order(order)
    return result.id
