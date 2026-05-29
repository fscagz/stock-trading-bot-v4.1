# test_broker.py

from broker_alpaca import get_account_info, get_open_positions, submit_market_order

print("=== ACCOUNT INFO ===")
info = get_account_info()
for k, v in info.items():
    print(f"{k}: {v}")

print("\n=== CURRENT POSITIONS ===")
positions = get_open_positions()
if not positions:
    print("No open positions.")
else:
    for symbol, qty in positions.items():
        print(f"{symbol}: {qty} shares")

# Uncomment to submit a test trade:
print("\n=== SUBMITTING ORDER ===")
order_id = submit_market_order("AAPL", 1, "buy")
print(f"Order submitted. ID: {order_id}")
