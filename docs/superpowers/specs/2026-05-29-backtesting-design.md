# V4 Backtesting System — Design Spec

## Goal

Replay the V4 momentum strategy against historical data to validate correctness on known events (targeted mode) and measure statistical performance across many days (statistical mode). Reuses all existing V4 logic unchanged — only the data source and order execution differ.

---

## Architecture

Four new modules added to `bot/backtest/`. The V3 backtest files (`engine.py`, `metrics.py`, etc.) are left untouched.

```
bot/backtest/
├── bar_fetcher.py         # fetches 1-min bars from Alpaca historical API; caches locally
├── candidate_screener.py  # daily bar screen — identifies Stage 1 movers per date
├── simulator.py           # replay engine — drives existing indicators/validator/manager
├── backtest_metrics.py    # computes summary stats from a list of TradeRecords
└── __main__.py            # CLI entry point

backtest_results/
├── cache/                 # 1-min bar JSON cache keyed by {SYMBOL}_{DATE}.json
├── trades_*.csv           # per-run closed trade log (same columns as live trade_logger)
└── summary_*.csv          # per-run aggregate metrics
```

---

## Components

### CandidateScreener (`candidate_screener.py`)

Identifies which symbols to replay on a given trading day, replacing the live scanner.

**Interface:**
```python
class CandidateScreener:
    def __init__(self, config: V4Config) -> None: ...
    def candidates_for_date(self, date: date) -> List[str]: ...
```

**Logic:** For each date, fetches daily OHLCV for a broad universe via `get_daily_batch` (Yahoo Finance, already in codebase). Filters symbols where:

```
(day_high - prev_close) / prev_close >= stage1_min_price_change_pct
AND day_close >= stage1_min_price
```

This is the same Stage 1 filter as the live scanner, applied to the full session's high — any stock that *reached* the threshold at any point during the day becomes a candidate.

Universe: all symbols in the most recent Alpaca assets list, filtered to NYSE/NASDAQ/AMEX and pure alphabetic tickers (same rules as live). The assets list is fetched once and cached for the run.

---

### BarFetcher (`bar_fetcher.py`)

Fetches and caches 1-minute historical bars from Alpaca's free IEX data feed.

**Interface:**
```python
class BarFetcher:
    def __init__(self, api_key: str, secret_key: str, cache_dir: str = "backtest_results/cache") -> None: ...
    def fetch(self, symbol: str, date: date) -> List[Bar]: ...
```

**Logic:**
- Cache key: `{cache_dir}/{SYMBOL}_{DATE}.json`
- On cache hit: deserialise and return
- On cache miss: `GET https://data.alpaca.markets/v2/stocks/bars` with `timeframe=1Min`, `feed=iex`, handles pagination via `next_page_token`
- Filters returned bars to regular session only: 09:30–16:00 ET
- Returns `List[Bar]` (the existing `Bar` dataclass)
- Symbols with no IEX data (returns empty list) are silently skipped

---

### Simulator (`simulator.py`)

Replays 1-min bars through all existing V4 components and produces `TradeRecord`s.

**Interface:**
```python
@dataclass
class BacktestResult:
    trades: List[TradeRecord]
    equity_curve: List[tuple[datetime, float]]   # (timestamp, equity)

class Simulator:
    def __init__(self, config: V4Config, initial_equity: float, slippage_pct: float = 0.001) -> None: ...
    def run_day(self, date: date, bars_by_symbol: Dict[str, List[Bar]], baseline_volumes: Dict[str, float]) -> BacktestResult: ...
```

**Replay logic per day:**

1. Instantiate fresh `ATRIndicator`, `VWAPIndicator`, `MomentumValidator`, `PositionManager`, `PortfolioState` (equity carries over between days in a multi-day run).
2. Merge all symbols' bars into a single list sorted by timestamp.
3. For each bar:
   - Update ATR and VWAP for the bar's symbol.
   - If symbol has an open position: run `PositionManager.on_bar()`. On exit instruction, simulate fill (see below) and complete the `TradeRecord`.
   - Otherwise: run `MomentumValidator.validate()`. On signal, simulate entry fill and open a `TradeRecord`.
4. At 15:25 ET bar: run `should_hold_overnight()`. If false, close position at bar close.
5. Return all closed `TradeRecord`s plus the equity curve.

**Execution simulation:**

| Event | Fill price |
|---|---|
| Entry (Stage 2 triggers on bar N) | Next bar's open + `slippage_pct` |
| Stop hit (`bar.low ≤ stop_price`) | `stop_price` (mid-bar trigger) |
| VWAP break / volume collapse / structure break | `bar.close` |
| EOD close | `bar.close` at 15:25 bar |

If no next bar exists after an entry signal (e.g., last bar of the day), the entry is skipped.

**Baseline volume:** passed in from the caller, computed the same way as the live Watchlist: `get_daily(symbol, period="1mo")["volume"].tail(20).mean() / 390`.

---

### BacktestMetrics (`backtest_metrics.py`)

Computes aggregate statistics from a list of `TradeRecord`s.

**Interface:**
```python
def compute_metrics(trades: List[TradeRecord], initial_equity: float) -> dict: ...
```

**Metrics returned:**

| Key | Description |
|---|---|
| `total_trades` | Count of closed trades |
| `win_rate` | Fraction with `pnl > 0` |
| `total_pnl` | Sum of all P&L |
| `avg_pnl_per_trade` | Mean P&L |
| `avg_winner` | Mean P&L on winning trades |
| `avg_loser` | Mean P&L on losing trades |
| `max_drawdown` | Largest peak-to-trough equity drop (absolute $) |
| `avg_hold_minutes` | Mean `(exit_time - entry_time).total_seconds() / 60` |
| `exit_reasons` | Dict of `{reason: count}` |

---

### CLI (`__main__.py`)

**Usage:**
```bash
# Statistical mode
python3.13 -m bot.backtest --start 2025-01-01 --end 2025-05-01

# Targeted mode
python3.13 -m bot.backtest --symbol ASTC --date 2021-12-10

# With slippage override
python3.13 -m bot.backtest --start 2025-01-01 --end 2025-05-01 --slippage 0.002
```

**Flow:**
1. Parse args; derive date range (targeted mode: single date, single symbol).
2. Load Alpaca credentials from `.env`.
3. For each trading day in range:
   a. `CandidateScreener.candidates_for_date(date)` — get symbol list.
   b. For each candidate: compute baseline volume via `get_daily`; `BarFetcher.fetch(symbol, date)`.
   c. `Simulator.run_day(date, bars_by_symbol, baseline_volumes)` — get trades.
4. Collect all `TradeRecord`s across all days.
5. `compute_metrics(all_trades, initial_equity)` — compute summary.
6. Write `backtest_results/trades_{start}_{end}.csv` and `backtest_results/summary_{start}_{end}.csv`.
7. Print summary table to terminal.

**Initial equity:** read from Alpaca account at startup (same as live bot), so the position sizing is realistic.

---

## Error Handling

- Symbols with no IEX bar data for a given date: skipped silently (logged at DEBUG).
- Yahoo Finance failures for baseline volume: baseline defaults to 0, symbol skipped for entry (validator returns False when baseline ≤ 0).
- Alpaca API errors for bar fetch: logged as WARNING, symbol skipped for that date.
- Days with zero candidates (weekend, holiday, no movers): skipped, logged at INFO.

---

## Testing

| Test file | Covers |
|---|---|
| `testing/test_candidate_screener.py` | Stage 1 filter logic with mocked `get_daily_batch` |
| `testing/test_bar_fetcher.py` | Cache hit/miss, bar parsing, session filter, pagination |
| `testing/test_simulator.py` | Entry fill on next bar, stop hit at stop price, EOD close, equity curve |
| `testing/test_backtest_metrics.py` | Win rate, max drawdown, exit reason counts |

No integration tests that hit real APIs.

---

## What Is Not In Scope

- Short selling (strategy is long-only)
- Pre/after-market bars
- Slippage modelling beyond a fixed percentage
- Walk-forward or parameter optimisation (separate concern)
- Visual P&L charts (CSV output is sufficient; user can chart in Excel/Sheets)
