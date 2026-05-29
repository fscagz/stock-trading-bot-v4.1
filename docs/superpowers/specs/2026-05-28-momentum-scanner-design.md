# Stock Trading Bot V4 — Momentum Scanner Design Spec

**Date:** 2026-05-28
**Status:** Approved

---

## 1. Overview

V4 is a momentum-based stock trading bot that detects stocks experiencing abnormal price movement combined with unusually high trading activity, then participates in the move while momentum is alive and exits when it deteriorates.

It is a standalone system, cloned from V3 with unnecessary components removed and a new momentum-specific core built in their place. V3 remains unchanged.

### Core Philosophy

The market often reveals unusual opportunities through momentum before the broader public fully reacts. Rather than predicting fundamentals or earnings, the system identifies stocks that are *already* entering a high-momentum state — measurable through price acceleration combined with significant volume expansion. These conditions create self-reinforcing momentum as more participants notice and join the move.

Entry requires confirmation, not just detection. A brief spike on low liquidity is not a signal. Sustainable momentum shows up in both price acceleration and elevated participation simultaneously.

Risk management is treated as equally important as the entry signal. Momentum stocks can reverse as fast as they rise.

---

## 2. What V4 Is and Is Not

**Is:**
- A momentum/breakout scanner and trader
- Long only (initially; short can be added later)
- Flexible hold — intraday by default, overnight if momentum is still intact at close
- Dynamic universe — discovers any US equity in real time, no static ticker list

**Is not:**
- A fundamental or value investor
- A high-frequency trader (minimum hold is minutes, not milliseconds)
- A replacement for V2 or V3

---

## 3. System Architecture

### Pipeline

```
┌─────────────────────────────────────────────────────┐
│  MARKET SCANNER (every 30s)                         │
│  Alpaca top movers + most actives → deduplicate     │
│  → apply Stage 1 filter                             │
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
│  Stage 2: price acceleration + relative volume      │
│  + buying pressure. All three must be true.         │
└─────────────────────┬───────────────────────────────┘
                      │ entry signal
┌─────────────────────▼───────────────────────────────┐
│  EXECUTION ENGINE                                   │
│  PortfolioState.can_enter() → compute position      │
│  size → place limit order → on fill: submit         │
│  broker-level hard stop order                       │
└─────────────────────┬───────────────────────────────┘
                      │ open position
┌─────────────────────▼───────────────────────────────┐
│  POSITION MANAGER (flexible hold)                   │
│  Per-bar: run exit logic (3 layers)                 │
│  At 15:25: evaluate overnight hold decision         │
└─────────────────────┬───────────────────────────────┘
                      │ exit signals / order updates
┌─────────────────────▼───────────────────────────────┐
│  RISK MANAGEMENT                                    │
│  KillSwitch, PortfolioState, ATR-based sizing       │
│  (all retained from V3)                             │
└─────────────────────────────────────────────────────┘
```

### Design Principle: Loose Coupling

The scanner and execution engine are intentionally decoupled:
- **Scanner** answers: "what stocks are interesting right now?"
- **Execution engine** answers: "should we trade this specific stock right now?"

The scanner has no knowledge of open positions or trade decisions. The execution engine has no knowledge of how a stock was discovered. They communicate only through the candidate watchlist.

---

## 4. Folder Structure

```
stock-trading-bot-v4/
├── bot/
│   ├── scanner/
│   │   ├── __init__.py
│   │   ├── market_scanner.py     # polls Alpaca top movers + most actives every 30s
│   │   └── watchlist.py          # manages candidate symbols and WebSocket subscriptions
│   ├── momentum/
│   │   ├── __init__.py
│   │   └── validator.py          # Stage 2 entry confirmation logic
│   ├── positions/
│   │   ├── __init__.py
│   │   └── manager.py            # per-bar exit logic, overnight hold decision
│   ├── intraday/                 # retained from V3
│   │   ├── indicators/
│   │   │   ├── atr.py            # ATR(14) on streaming bars
│   │   │   └── vwap.py           # intraday VWAP, resets daily per symbol
│   │   └── risk/
│   │       ├── sizing.py         # ATR-based position sizing
│   │       ├── portfolio.py      # PortfolioState: heat, sector cap, positions, cooldown
│   │       └── kill_switch.py    # daily drawdown + slippage halt; loss cooldown
│   ├── data/
│   │   ├── __init__.py
│   │   ├── stream.py             # BarStream: Alpaca WebSocket 1-min bars
│   │   └── daily_loader.py       # historical OHLCV for baseline avg volume computation
│   ├── broker_alpaca.py          # Alpaca order placement (extended for stop + limit orders)
│   ├── backtest/                 # simulation engine (retained from V3)
│   ├── monitoring/               # trade logging, performance tracking (retained from V3)
│   └── config.py                 # all V4 thresholds and parameters
├── testing/
├── scripts/
├── requirements.txt
└── README.md
```

---

## 5. What Is Retained from V3 vs. Removed

### Retained

| Component | Role in V4 |
|-----------|------------|
| `data/stream.py` (BarStream) | WebSocket 1-min bar feed — core data pipe for all symbol monitoring |
| `intraday/indicators/atr.py` | Computes ATR(14) on streaming bars — used for stop/target distance and momentum deterioration |
| `intraday/indicators/vwap.py` | Intraday VWAP — used as a momentum deterioration exit signal |
| `intraday/risk/sizing.py` | ATR-based position sizing — shares = risk_dollars / stop_distance |
| `intraday/risk/portfolio.py` | PortfolioState: tracks positions, heat, sector cap, max positions, cooldown gate |
| `intraday/risk/kill_switch.py` | Daily drawdown halt, slippage spike halt, consecutive loss cooldown |
| `broker_alpaca.py` | Alpaca order placement — extended in V4 for stop and limit orders |
| `backtest/` | Historical simulation for validating entry/exit criteria before live |
| `data/daily_loader.py` | Historical OHLCV — used at startup to compute 20-day avg volume baseline per symbol |
| `monitoring/` | Trade logging and performance tracking |

### Removed

| Component | Reason |
|-----------|--------|
| `intraday/signals/` (VWAP continuation, event, sentiment) | Replaced by momentum validator |
| `intraday/data/universe.py` / `universe_loader.py` | Replaced by dynamic scanner — no static universe |
| `intraday/data/event_calendar.py` | Not relevant to momentum strategy |
| `intraday/data/news_stream.py` | Not relevant |
| `intraday/ml/` | No ML layer in V4 initially |
| `features/`, `models/` | Not relevant |
| Hard 15:30 intraday close loop | Replaced by flexible overnight hold decision |

---

## 6. Screening Engine

**Sources:**
- Alpaca `/v1beta1/screener/stocks/movers` — top gainers by % price change
- Alpaca `/v1beta1/screener/stocks/most-actives` — most active by dollar volume

**Frequency:** Every 30 seconds

**Logic:**
1. Fetch both lists
2. Union and deduplicate
3. Apply Stage 1 filter (see below)
4. Add passing symbols to candidate watchlist; subscribe to BarStream

A stock appearing on **both** lists (big % move + high participation) is a stronger candidate than one appearing on only one.

No static ticker list — any US equity can be discovered.

---

## 7. Entry Logic (Two-Stage Momentum Validation)

### Stage 1 — Scanner Filter (applied every 30s to movers/most-actives results)

| Filter | Threshold | Reasoning |
|--------|-----------|-----------|
| Price up on the day | ≥ 5% | Below 5% is normal daily noise for most stocks |
| Price | ≥ $0.50 (no upper limit) | Catches low-float movers; fractional shares handle high prices |
| On both movers + most-actives | Preferred, not required | Dual-list appearance indicates stronger participation |

### Stage 2 — Momentum Validation (applied per 1-min bar on stream)

| Condition | Threshold | Reasoning |
|-----------|-----------|-----------|
| Short-term rate of change | Close ≥ 3% above close 5 bars ago | Measures acceleration, not just total daily gain |
| Relative volume | Current bar volume ≥ 4× 20-day per-minute average | Baseline = 20-day avg daily volume ÷ 390 (trading minutes/day) |
| Buying pressure | Close in top 25% of bar's high-low range | Buyers in control at bar close |

All three Stage 2 conditions must be simultaneously true. Two-of-three is insufficient.

---

## 8. Execution

On a confirmed Stage 2 signal:

1. **Portfolio check** — `PortfolioState.can_enter()`: kill switch, cooldown, max positions, heat cap, sector cap
2. **Size position** — `compute_position_size(equity, atr, entry_price, config)`
3. **Place entry** — Limit order slightly above current ask (aggressive limit)
4. **Cancel if unfilled** — Cancel and re-evaluate if not filled within 30 seconds
5. **On fill** — Immediately submit broker-level hard stop order to Alpaca

---

## 9. Exit Logic (Three-Layer System)

### Layer 1 — Hard Stop (broker-level)
- Price: 1.5× ATR(14) below entry
- Submitted to Alpaca as a native stop order immediately after fill
- Executes even if bot software is down
- Trigger: market exit, no discretion

### Layer 2 — Trailing Stop (software-managed, updates broker order)
- Activates once position is profitable
- Level: 2× ATR below the highest close seen since entry
- Only moves up, never down
- As price makes new highs, software cancels and resubmits the broker stop order at the new trailing level
- Trigger: market exit

### Layer 3 — Momentum Deterioration (software-monitored, fires before stops)
Graceful exits while liquidity is still good:

| Signal | Condition | Action |
|--------|-----------|--------|
| VWAP break | Close below VWAP on bar volume ≥ 2× per-minute average | Limit exit |
| Structure break | Lower high + lower low on 2 consecutive bars | Reduce or close |
| Volume collapse | Current bar volume < 0.5× entry bar volume | Close |
| Rate of change reversal | 5-bar rate of change turns negative | Close |

### Overnight Hold Decision (evaluated at 15:25)

Hold overnight if ALL of:
- Price is above VWAP
- Volume in the last 30 minutes ≥ 2× per-minute baseline
- No Layer 3 momentum deterioration signals have fired

Otherwise close before 15:30.

---

## 10. Broker Order Extensions (broker_alpaca.py)

V3's broker module only supports market orders. V4 requires:

| Method | Purpose |
|--------|---------|
| `submit_limit_order(symbol, qty, side, limit_price)` | Entry orders; graceful momentum exits |
| `submit_stop_order(symbol, qty, stop_price)` | Broker-level hard stop after fill |
| `cancel_order(order_id)` | Cancel hard stop before resubmitting trailing level |
| `submit_market_order(symbol, qty, side)` | Already exists; used for hard stop hits |

---

## 11. Risk Parameters

### Carried Over from V3

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

### Adjusted for V4

| Parameter | V3 | V4 | Reason |
|-----------|----|----|--------|
| Sector cap | 2 positions | 5 positions | Momentum events cluster in sectors; tight cap misses correlated opportunities |
| Min price | $5.00 | $0.50 | Explicitly targeting low-float movers |
| Max price | $500 | No limit | Momentum happens at any price; fractional shares handle sizing |
| Spread filter | < 0.1% of mid | < 1.0% of mid | Sub-$5 stocks have wider spreads; 0.1% excludes the target category |

---

## 12. Open Questions / Future Work

- **Short selling** — intentionally excluded from V4 initial build; can be added once long-only is validated
- **ML scoring layer** — not in V4 initially; could be added after sufficient trade log data is collected
- **Polygon.io upgrade** — if Alpaca's screener endpoints prove insufficient (rate limits, missing stocks), Polygon.io is the natural upgrade path for market-wide streaming
- **Backtesting thresholds** — the specific numbers in Sections 7 and 9 should be validated against historical intraday data before going live; treat them as starting points
