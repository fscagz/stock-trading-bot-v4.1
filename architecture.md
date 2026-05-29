# Stock Trading Bot V4 — Architecture

## Philosophy

React to measurable signs that a stock is already entering a high-momentum state.
Do not predict — detect. Combine price acceleration, relative volume, and continuation
behavior to enter early in strong moves and exit when momentum deteriorates.

- **Long only** (initially)
- **Flexible hold** — intraday by default, overnight if momentum remains intact at close
- **Two-stage entry** — scanner discovers, validator confirms
- **Scanner and execution are loosely coupled** — scanner answers "what's interesting?", execution engine answers "should we trade this right now?"

---

## Pipeline

```
┌─────────────────────────────────────────────────────┐
│  MARKET SCANNER (every 30s)                         │
│  Alpaca top movers + most actives → deduplicate     │
│  → apply basic price/volume filter                  │
└─────────────────────┬───────────────────────────────┘
                      │ candidate symbols
┌─────────────────────▼───────────────────────────────┐
│  CANDIDATE WATCHLIST                                │
│  Tracks "in play" stocks                            │
│  Subscribes/unsubscribes symbols on BarStream       │
└─────────────────────┬───────────────────────────────┘
                      │ real-time 1-min bars
┌─────────────────────▼───────────────────────────────┐
│  REALTIME STREAM (BarStream)                        │
│  Feeds bars to: ATR, VWAP, Momentum Validator       │
└─────────────────────┬───────────────────────────────┘
                      │ confirmed momentum signal
┌─────────────────────▼───────────────────────────────┐
│  MOMENTUM VALIDATOR                                 │
│  Price acceleration + relative volume + buying      │
│  pressure. Two-stage gate before entry signal.      │
└─────────────────────┬───────────────────────────────┘
                      │ entry signal
┌─────────────────────▼───────────────────────────────┐
│  EXECUTION ENGINE                                   │
│  PortfolioState.can_enter() → size position →       │
│  place limit order → record fill                    │
└─────────────────────┬───────────────────────────────┘
                      │ open position
┌─────────────────────▼───────────────────────────────┐
│  POSITION MANAGER (flexible hold)                   │
│  Monitors each bar. Runs exit logic.                │
│  At 15:30: hold overnight if momentum intact,       │
│  otherwise close.                                   │
└─────────────────────┬───────────────────────────────┘
                      │ exit signals / filled orders
┌─────────────────────▼───────────────────────────────┐
│  RISK MANAGEMENT                                    │
│  KillSwitch, PortfolioState, ATR-based sizing       │
└─────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
stock-trading-bot-v4/
├── bot/
│   ├── scanner/
│   │   ├── market_scanner.py     # polls Alpaca top movers + most actives every 30s
│   │   └── watchlist.py          # manages candidate symbols, WebSocket subscriptions
│   ├── momentum/
│   │   └── validator.py          # price acceleration + relative volume + buying pressure
│   ├── positions/
│   │   └── manager.py            # per-bar exit logic, overnight hold decision at 15:30
│   ├── intraday/                 # retained from V3
│   │   ├── indicators/
│   │   │   ├── atr.py            # ATR(14) on streaming bars — used for stop/target sizing
│   │   │   └── vwap.py           # intraday VWAP — used as momentum deterioration signal
│   │   └── risk/
│   │       ├── sizing.py         # ATR-based position sizing (risk_per_trade % of equity)
│   │       ├── portfolio.py      # PortfolioState: heat, sector cap, max positions, cooldown
│   │       └── kill_switch.py    # daily drawdown + slippage halt; consecutive loss cooldown
│   ├── data/
│   │   ├── stream.py             # BarStream: Alpaca WebSocket 1-min bars (retained from V3)
│   │   └── daily_loader.py       # historical OHLCV for baseline avg volume (retained from V3)
│   ├── broker_alpaca.py          # Alpaca order placement — extended for limit + OCO orders
│   ├── backtest/                 # simulation engine (retained from V3)
│   ├── monitoring/               # trade logging, performance tracking (retained from V3)
│   └── config.py                 # V4 config (thresholds, risk params, scanner intervals)
├── testing/
├── scripts/
├── requirements.txt
└── README.md
```

---

## Entry Logic (Momentum Validator)

Two-stage gate. Both stages must pass before an order is placed.

**Stage 1 — Scanner filter (every 30s, broad)**

| Filter | Threshold | Reasoning |
|--------|-----------|-----------|
| Price up on the day | ≥ 5% | Below 5% is normal daily noise for most stocks |
| Price | ≥ $0.50 (no upper limit) | Catches low-float movers like ASTC; fractional shares handle high prices |
| On both movers + most-actives | Preferred, not required | Appearing on both lists signals stronger participation |

**Stage 2 — Momentum validation (per 1-min bar, on stream)**

| Condition | Threshold | Reasoning |
|-----------|-----------|-----------|
| Short-term rate of change | Close ≥ 3% above close 5 bars ago | Measures acceleration, not just total daily gain |
| Relative volume | Current bar volume ≥ 4× 20-day per-minute average | Strong participation; baseline = 20-day avg daily volume ÷ 390 |
| Buying pressure | Close in top 25% of bar's high-low range | Buyers in control at bar close |

All three Stage 2 conditions must be simultaneously true.

---

## Exit Logic (Position Manager)

Exits are moderately aggressive — protect capital and lock in gains over maximizing the top.

Three layers work together:

**Layer 1 — Hard stop (broker-level, immediate)**
- Set at entry: 1.5× ATR(14) below entry price
- Submitted to Alpaca as a native stop order immediately after fill confirmation
- Executes even if the bot software crashes — position is always protected
- If price hits this → market exit, no discretion

**Layer 2 — Trailing stop (software-managed, updates broker order)**
- Activates once the trade is profitable
- Trails at 2× ATR below the highest close seen since entry
- Only moves up, never down
- As price makes new highs, software cancels and resubmits the broker stop order at the new level
- Gives strong runners room while locking in gains

**Layer 3 — Momentum deterioration (software-monitored, fires before stops)**
Graceful exits while liquidity is still good — fire before either stop is hit:

| Signal | Condition | Action |
|--------|-----------|--------|
| VWAP break | Close below VWAP on volume ≥ 2× average | Limit exit |
| Structure break | Lower high + lower low on 2 consecutive bars | Reduce or close |
| Volume collapse | Current bar volume < 0.5× entry bar volume | Close |
| Rate of change reversal | 5-bar rate of change turns negative | Close |

**Overnight hold decision (evaluated at 15:25 each day)**

Hold overnight if ALL of:
- Price is above VWAP
- Volume in last 30 min still elevated (≥ 2× per-minute baseline)
- No momentum deterioration signals have fired

Otherwise close before 15:30.

**Broker order requirements (extensions to broker_alpaca.py):**
- `submit_stop_order(symbol, qty, stop_price)` — place hard stop at broker
- `cancel_order(order_id)` — cancel existing stop before replacing
- `submit_limit_order(symbol, qty, side, limit_price)` — graceful momentum exits
- `submit_market_order(symbol, qty, side)` — already exists; used for hard stop hits

---

## Risk Parameters

**Carried over from V3 unchanged:**

| Parameter | Value |
|-----------|-------|
| Risk per trade | 0.5% of equity |
| Hard stop | 1.5× ATR(14) from entry |
| Trailing stop | 2× ATR below highest close since entry |
| Max position size | 5% of equity |
| Max daily drawdown | -2% of equity (kill switch) |
| Max open positions | 5 simultaneous |
| Portfolio heat cap | Total open risk ≤ 3% of equity |
| Consecutive loss cooldown | 30-min pause after 3 consecutive losses |

**Adjusted for V4:**

| Parameter | V3 | V4 | Reason |
|-----------|----|----|--------|
| Sector cap | 2 positions | 5 positions | Momentum events cluster in sectors; being too tight misses correlated opportunities |
| Min price | $5.00 | $0.50 | Explicitly targeting low-float movers |
| Max price | $500 | No limit | Momentum happens at any price; fractional shares handle sizing |
| Spread filter | < 0.1% of mid | < 1.0% of mid | Sub-$5 stocks have wider spreads; 0.1% would exclude the category we want |

---

## Screening Engine

- **Source:** Alpaca `/v1beta1/screener/stocks/movers` + `/v1beta1/screener/stocks/most-actives`
- **Frequency:** Every 30 seconds
- **Logic:** Union both lists, deduplicate. A stock on both lists (big % move AND high participation) is a stronger candidate.
- **No static ticker list** — discovers any US equity dynamically

---

## What Is Retained from V3

| Component | Why Kept |
|-----------|----------|
| `stream.py` (BarStream) | WebSocket 1-min bar feed — core data pipe |
| `indicators/atr.py` | Stop/target sizing; momentum deterioration detection |
| `indicators/vwap.py` | Exit signal — VWAP break indicates momentum loss |
| `risk/sizing.py` | ATR-based position sizing tied to equity % risk |
| `risk/portfolio.py` | Portfolio heat, sector cap, max positions gate |
| `risk/kill_switch.py` | Daily drawdown halt, slippage spike halt, loss cooldown |
| `broker_alpaca.py` | Alpaca order placement (extended for limit + OCO) |
| `backtest/` | Validates entry/exit criteria historically before live |
| `data/daily_loader.py` | Historical OHLCV for computing baseline avg volume |
| `monitoring/` | Trade logging and performance tracking |

## What Is Removed from V3

| Component | Why Removed |
|-----------|-------------|
| `intraday/signals/` (VWAP continuation, event, sentiment) | Replaced by momentum validator |
| `intraday/data/universe.py` / `universe_loader.py` | Replaced by dynamic scanner |
| `intraday/data/event_calendar.py`, `news_stream.py` | Not relevant to momentum strategy |
| `intraday/ml/` | No ML layer in V4 initially |
| `features/`, `models/` | Not relevant |
| Hard 15:30 intraday close loop | Replaced by flexible overnight hold logic |
