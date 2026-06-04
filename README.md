# Stock Trading Bot V4.1

A momentum-detection trading bot built on real-time market data from **IBKR (IB Gateway)**. It scans the entire US equity universe in real time, identifies stocks entering high-momentum states, enters with ATR-based position sizing, and exits via a three-layer system. A full backtesting engine lets you replay the strategy against historical 1-minute bars before going live.

Market data (live streaming and historical bars) comes from IB Gateway. Order execution, account info, and the ETB list use [Alpaca Markets](https://alpaca.markets).

---

## Philosophy

> React to measurable signs that a stock is already entering a high-momentum state. Do not predict — detect.

The bot combines price acceleration, relative volume, and buying pressure to enter early in strong moves and exit before momentum fully deteriorates. It does not use ML, news sentiment, or any static watchlist — every candidate is discovered dynamically.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  MARKET SCANNER (every 30s)                         │
│  Full US equity universe → IEX snapshots            │
│  → Stage 1 filter (price % change + min price)      │
└─────────────────────┬───────────────────────────────┘
                      │ candidate symbols
┌─────────────────────▼───────────────────────────────┐
│  CANDIDATE WATCHLIST                                │
│  Tracks "in-play" stocks                            │
│  Subscribes / unsubscribes symbols on BarStream     │
└─────────────────────┬───────────────────────────────┘
                      │ real-time 1-min bars
┌─────────────────────▼───────────────────────────────┐
│  REALTIME STREAM (BarStream)                        │
│  IBKR reqRealTimeBars (5-sec) →                     │
│  MinuteBarAggregator → 1-min Bar objects            │
│  Feeds bars to: ATR(14), VWAP, Momentum Validator   │
└─────────────────────┬───────────────────────────────┘
                      │ confirmed momentum signal
┌─────────────────────▼───────────────────────────────┐
│  MOMENTUM VALIDATOR (Stage 2)                       │
│  Rate-of-change + relative volume + buying pressure │
└─────────────────────┬───────────────────────────────┘
                      │ entry signal
┌─────────────────────▼───────────────────────────────┐
│  EXECUTION ENGINE                                   │
│  PortfolioState.can_enter() → ATR-size position →   │
│  place limit order (Alpaca) → confirm fill → stop   │
└─────────────────────┬───────────────────────────────┘
                      │ open position
┌─────────────────────▼───────────────────────────────┐
│  POSITION MANAGER                                   │
│  Per-bar exit logic (3 layers)                      │
│  At 15:25 ET: hold overnight or close               │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│  RISK MANAGEMENT                                    │
│  KillSwitch · PortfolioState · ATR-based sizing     │
└─────────────────────────────────────────────────────┘
```

---

## Setup

### Requirements

- Python 3.11+
- An [Alpaca Markets](https://alpaca.markets) account (paper or live) — for order execution, account info, and ETB list
- **IB Gateway** running locally — for market data (live streaming and backtest bar fetching)

```bash
pip install -r requirements.txt
```

### IB Gateway

Start IB Gateway before running the bot or any backtest. The default ports are:

| Mode | Port |
|------|------|
| Paper trading | 4002 |
| Live trading | 4001 |

### Environment

Create a `.env` file in the project root:

```
# Alpaca — order execution, account info, ETB list
APCA_API_KEY_ID=your_key
APCA_API_SECRET_KEY=your_secret
APCA_API_BASE_URL=https://paper-api.alpaca.markets   # or https://api.alpaca.markets for live

# IBKR — market data (live stream + backtest bar fetch)
IBKR_HOST=127.0.0.1
IBKR_PORT=4002                  # 4002 paper, 4001 live
IBKR_CLIENT_ID_STREAM=1         # client ID for the live bar stream
IBKR_CLIENT_ID_FETCHER=2        # client ID for backtest bar fetching (separate connection)
```

---

## Usage

### Run the Live Bot (Long / Catalyst Mode) with Dashboard

```bash
python -m bot.main
```

The bot loads `make_long_config()` at startup. It scans the full US equity universe every 30 seconds, watches for catalyst-driven moves ≥15% above prior close, and enters on confirmed momentum. A web dashboard is available at `http://localhost:8080` (override port with `DASHBOARD_PORT`). Trades are written to `logs/` as they close.

### Run the Live Bot (Long + Short Combined)

```bash
python -m bot.live
# or with reduced risk:
python -m bot.live --risk-scale 0.5
```

Runs the full dual-portfolio `LiveRunner` (simultaneous long and short strategies with independent heat budgets). Does not launch the dashboard.

### Backtest — 2025-2026 Bull Year

Bar data is fetched from IBKR and cached automatically on first run. Subsequent runs use the local cache and are fast. IB Gateway must be running for the initial fetch.

```bash
python3 -m bot.backtest --long --start 2025-06-01 --end 2026-05-28 --risk-scale 2.0
```

Expected output (at 2× risk scale, equity at time of run):

| Metric | Result |
|--------|--------|
| Trades | 28 |
| Win rate | 53.6% |
| Total P&L | +$33,345 |
| Max drawdown | $7,300 |

### Backtest — 2022 Bear Year

Build the bar cache first (one-time, ~20-30 min, IB Gateway must be running):

```bash
python3 build_2022_cache.py
```

Once complete, run the backtest normally:

```bash
python3 -m bot.backtest --long --start 2022-01-03 --end 2022-12-30 --risk-scale 2.0
```

Expected output (at 2× risk scale, equity at time of run):

| Metric | Result |
|--------|--------|
| Trades | 139 |
| Win rate | 36.7% |
| Total P&L | +$199 (approx. breakeven) |
| Max drawdown | $30,253 |

The strategy is designed to be market-regime agnostic: it enters only when a genuine catalyst triggers abnormal volume and price acceleration, so it naturally takes fewer and higher-quality trades in bear years rather than going short or shutting off.

---

## Entry Logic

Entry requires both stages to pass simultaneously.

### Stage 1 — Market Scanner (every 30 seconds)

Fetches real-time IEX snapshots for every active NYSE/NASDAQ/AMEX equity and applies a broad filter:

| Filter | Default | Notes |
|--------|---------|-------|
| Price change vs prior close | ≥ 5% | Catches stocks already in motion |
| Minimum price | ≥ $5.00 | Configurable; `make_long_config()` lowers to $2.00 |

Passing symbols are added to the watchlist and subscribed on the live bar stream.

### Stage 2 — Momentum Validator (per 1-minute bar)

All three conditions must be true on the same bar:

| Condition | Default Threshold | What it measures |
|-----------|------------------|-----------------|
| Rate of change | Close ≥ 3% above close 5 bars ago | Price acceleration, not just daily gain |
| Relative volume | Bar volume ≥ 4× 20-day per-minute average | Abnormal participation |
| Buying pressure | Close position in bar range within configured window | Buyers still in control at bar close |

Optional filters (disabled by default, configurable):

- `stage2_require_green_bar` / `stage2_require_red_bar` — entry bar must close green or red
- `stage2_volume_exhaustion_ratio` — entry volume must be below a fraction of the recent peak bar
- `stage2_require_volume_decline` — entry volume must be less than the immediately prior bar
- `stage2_max_spike_age_bars` / `stage2_min_spike_age_bars` — restrict entries to a window around when the stock first became eligible
- `stage2_min_gap_pct` + `stage2_max_gap_entry_minutes` — gap-up fade filters

### Confidence-Based Position Sizing

After Stage 2 passes, a 0–1 confidence score is computed from how far each signal exceeds its minimum. The score maps to one of four risk multiplier tiers, allowing stronger signals to receive proportionally larger positions. All multipliers default to 1.0 (uniform sizing); the `make_long_config()` profile uses 1×/2×/4×/8× tiers.

### Entry Execution

1. A limit order is placed at `close × (1 + limit_offset_pct)`.
2. The bot waits up to `fill_timeout_seconds` for a fill.
3. On fill: a hard stop order is submitted immediately to the broker at `entry - stop_atr_multiple × ATR(14)`.

---

## Exit Logic

Three layers work in priority order on every bar.

### Layer 1 — Hard Stop (broker-level)

- Set at entry: `stop_atr_multiple × ATR(14)` below the fill price.
- Submitted to Alpaca as a native stop order — executes even if the bot software crashes.

### Layer 2 — Trailing Stop (software-managed)

- Activates once the breakeven trigger is reached (`breakeven_trigger_atr_multiple × ATR` in profit).
- Trails `trailing_stop_atr_multiple × ATR` below the highest close seen since entry.
- Only moves in the favorable direction. On each update the broker stop order is cancelled and resubmitted.

### Layer 3 — Momentum Deterioration (fires before stops when possible)

| Signal | Condition | Action |
|--------|-----------|--------|
| VWAP break | Close below VWAP on volume ≥ 2× average | Limit exit at close |
| Volume collapse | Bar volume < 0.5× entry bar volume | Market exit |
| Structure break | Lower high + lower low on 2 consecutive bars | Limit exit at close |

### Overnight Hold Decision (15:25 ET)

Hold overnight only if **all** of:
- Close ≥ VWAP
- Last bar volume ≥ 2× per-minute baseline

Otherwise close before 15:30.

### Short Positions

All exit logic mirrors the long logic with direction inverted: the trailing stop trails the lowest close upward, the VWAP break fires when price rises above VWAP on volume, and structure break detects higher high + higher low.

---

## Risk Management

### Portfolio Constraints

| Parameter | Default Value |
|-----------|--------------|
| Max open positions | 15 |
| Max sector positions | 5 |
| Max portfolio heat (total open risk) | 12% of equity |
| Risk per trade | 2% of equity |
| Max single position | 20% of equity |

### Kill Switch (`bot/intraday/risk/kill_switch.py`)

Halts all new entries for the session when triggered:

1. **Daily drawdown** — P&L drops below `max_daily_drawdown` threshold.
2. **Slippage spike** — actual session slippage exceeds `slippage_kill_multiple × expected`.
3. **Consecutive loss cooldown** — after `consecutive_loss_trigger` losses in a row, trading pauses for `consecutive_loss_cooldown_minutes` (not a full halt).

### Position Sizing Formula (`bot/intraday/risk/sizing.py`)

```
shares = floor((equity × risk_per_trade) / (stop_atr_multiple × ATR))
shares = min(shares, floor((equity × max_position_pct) / entry_price))
```

---

## Configuration (`bot/config.py`)

`V4Config` is a dataclass with all tunable parameters. Key presets:

- **Default** (`V4Config()`) — short/fade mode defaults: min price $5, relative volume 4×, buying pressure in bottom 50% of bar range.
- **Long mode** (`make_long_config()`) — catalyst-driven longs: min price $2, relative volume 10×, buying pressure in top 15% of range, confidence tiers enabled (1×/2×/4×/8×).

---

## Folder Structure

```
stock-trading-bot-v4.1/
├── bot/
│   ├── main.py                    # live trading entry point (long mode + dashboard)
│   ├── config.py                  # V4Config + make_long_config()
│   ├── broker_alpaca.py           # Alpaca order placement, account info, ETB list
│   ├── trade_logger.py            # CSV trade log
│   ├── live/
│   │   ├── __main__.py            # entry point: python -m bot.live (long+short combined)
│   │   └── runner.py              # LiveRunner — dual long+short portfolio
│   ├── scanner/
│   │   ├── market_scanner.py      # IEX snapshot scanner (every 30s)
│   │   └── watchlist.py           # candidate set + IBKR stream subscriptions
│   ├── momentum/
│   │   └── validator.py           # Stage 2 validation + confidence score
│   ├── positions/
│   │   └── manager.py             # per-bar exit logic + overnight decision
│   ├── dashboard/
│   │   ├── server.py              # FastAPI/uvicorn server (localhost:8080)
│   │   ├── state.py               # shared dashboard state
│   │   └── templates.py           # HTML template
│   ├── intraday/
│   │   ├── indicators/
│   │   │   ├── atr.py             # ATR(14) on streaming bars
│   │   │   └── vwap.py            # intraday VWAP
│   │   ├── risk/
│   │   │   ├── sizing.py          # ATR-based position sizing
│   │   │   ├── portfolio.py       # heat / sector cap / max positions
│   │   │   └── kill_switch.py     # daily drawdown + slippage halt
│   │   └── data/
│   │       ├── stream.py          # IBKR reqRealTimeBars (5-sec) bar feed
│   │       └── aggregator.py      # MinuteBarAggregator: 5-sec → 1-min bars
│   ├── backtest/
│   │   ├── __main__.py            # backtest CLI
│   │   ├── simulator.py           # Simulator + CombinedSimulator
│   │   ├── bar_fetcher.py         # IBKR reqHistoricalData 1-min bars with local cache
│   │   ├── candidate_screener.py  # daily bar pre-screen for candidates
│   │   ├── news_filter.py         # catalyst detection via Alpaca news
│   │   ├── backtest_metrics.py    # trade-level metrics (win rate, P&L, etc.)
│   │   ├── metrics.py             # return-based metrics (Sharpe, CAGR, IC)
│   │   └── costs.py               # slippage + commission modeling
│   └── data/
│       └── daily_loader.py        # historical OHLCV for baseline volume
├── testing/                       # pytest suite
├── backtest_results/              # CSV output from backtests
├── requirements.txt
└── architecture.md
```

---

## Backtesting

The backtest engine replays the exact same entry/exit logic used live against historical 1-minute bars fetched from IBKR. Bars are fetched once and cached locally so reruns are fast. IB Gateway must be running for any cache miss.

IBKR paces historical data requests to ~6 per minute per connection; the fetcher enforces a 10-second inter-request delay to stay within limits. Use `IBKR_CLIENT_ID_FETCHER` (default client ID 2) so the backtest connection does not conflict with a running live stream.

### Modes

| Mode | Flag | Description |
|------|------|-------------|
| Short / fade | `--short` | Fade momentum spikes; news filter defaults to `exclude` (skip real catalyst runners) |
| Long / catalyst | `--long` | Ride high-magnitude catalyst moves; news filter defaults to `require` |
| Combined | `--both` | Run both strategies simultaneously with independent heat budgets on the same bar stream |

### Running a Backtest

**Single symbol, single day (targeted mode):**
```bash
python -m bot.backtest --symbol AAPL --date 2024-03-15
```

**Date range (statistical mode):**
```bash
python -m bot.backtest --start 2024-01-01 --end 2024-06-30
```

**Short mode over a date range:**
```bash
python -m bot.backtest --short --start 2024-01-01 --end 2024-06-30
```

**Long mode (catalyst required):**
```bash
python -m bot.backtest --long --start 2024-01-01 --end 2024-06-30
```

**Combined (short + long simultaneously):**
```bash
python -m bot.backtest --both --start 2024-01-01 --end 2024-06-30
```

**Additional options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--slippage` | `0.001` | Fill slippage fraction (0.1%) |
| `--etb-only` | off | Restrict short entries to easy-to-borrow symbols |
| `--risk-scale` | `1.0` | Scale `risk_per_trade` and `max_portfolio_heat` |
| `--news-mode` | auto | `exclude` / `require` / `ignore` — override news filter |

### Output

Results are written to `backtest_results/`:

- `trades_<prefix>.csv` — every closed trade: ticker, entry/exit time, prices, P&L, exit reason, portfolio heat at entry.
- `summary_<prefix>.csv` — aggregate metrics printed to stdout and saved.

### Metrics Reported

| Metric | Target |
|--------|--------|
| CAGR | > SPY benchmark |
| Sharpe ratio | > 0.7 |
| Sortino ratio | > 1.0 |
| Max drawdown | > −25% |
| Calmar ratio | > 0.5 |
| Hit rate | > 52% |
| IC mean | > 0.05 |
| IC t-stat | > 2.0 |

### How the Simulator Works

1. **Candidate screening** — `CandidateScreener` loads daily bars to find which symbols had a large enough move on each backtest date, applying the same Stage 1 filter as the live scanner.
2. **Bar fetching** — `BarFetcher` retrieves 1-minute bars for each candidate from IBKR (`reqHistoricalData`, regular trading hours). Bars are cached to disk; parallel fetching (16 threads) is used when all bars are already cached.
3. **Replay** — `Simulator.run_day()` merges all symbol bars chronologically and processes them one bar at a time through the exact same `MomentumValidator`, `PositionManager`, and `PortfolioState` instances used in live trading.
4. **Entry lag** — a symbol must appear on the previous bar before an entry is considered, mirroring the ~1-bar watchlist subscription delay in the live bot.
5. **Slippage** — entries fill at `open × (1 ± slippage_pct)` on the bar after the signal bar; stop fills add additional slippage.
6. **Overnight carry** — positions passing the 15:25 ET overnight check are carried into the next trading day.
7. **Combined mode** — `CombinedSimulator` runs short and long `PortfolioState` objects independently on the same bar stream; equity is shared and synced after every close.

---

## Testing

```bash
pytest testing/
```

The test suite covers the bar fetcher, candidate screener, simulator, momentum validator, position manager, indicators, broker extensions, and more.
