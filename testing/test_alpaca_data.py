from alpaca_trade_api.rest import REST, TimeFrame
import os

api = REST(
    os.getenv("APCA_API_KEY_ID"),
    os.getenv("APCA_API_SECRET_KEY"),
    base_url="https://api.alpaca.markets"
)

bars = api.get_bars("SPY", TimeFrame.Minute, limit=10, feed="sip").df

print(bars)
