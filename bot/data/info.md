Description of the data layer:

1.1 Daily OHLCV loader (bot/data/daily_loader.py)
get_daily(symbol, start, end, period) – Daily OHLCV for one symbol from yfinance; returns a DataFrame with date index and open, high, low, close, volume.
get_daily_batch(symbols, start, end, period) – Same for many symbols; returns dict[symbol -> DataFrame] (only symbols with data).
Standardized column names and date index (no timezone, normalized).
1.2 Investment universe (bot/data/universe.py)
get_tickers_sp500() – S&P 500 tickers from Wikipedia (User-Agent, . → -).
compute_adv_from_panel(symbol_to_df) – Average daily volume per symbol from daily DataFrames.
get_universe(source, min_adv, min_market_cap, ...) – Filtered universe: S&P 500, then by minimum ADV (and optional market cap).
get_universe_from_config() – Uses config (e.g. MIN_AVG_DAILY_VOLUME, MIN_MARKET_CAP).
should_refresh_universe(last_refresh_date, quarterly) – Whether to refresh the universe (e.g. new quarter).
1.3 Fundamentals (bot/data/fundamentals.py)
get_fundamentals(symbol) – One symbol: value (P/E, EV/EBITDA, FCF yield), quality (ROE, gross margin, debt-to-equity), growth (revenue, EPS), plus placeholder for earnings consistency; all from yfinance .info.
get_fundamentals_batch(symbols) – Same for many symbols; returns dict[symbol -> dict].
FUNDAMENTAL_KEYS – Canonical list of factor names for the feature engine.
1.4 Cache/store (bot/data/store.py)
Daily: save_daily(symbol, df) / load_daily(symbol) – CSV under {CACHE_DIR}/daily/{SYMBOL}.csv.
Fundamentals: save_fundamentals(symbol, data) / load_fundamentals(symbol) – JSON under {CACHE_DIR}/fundamentals/{SYMBOL}.json.
Staleness: is_daily_stale(symbol, max_age_days), is_fundamentals_stale(symbol, max_age_days) and cache_age_days(path) for controlling refetches.
1.5 Point-in-time (bot/data/point_in_time.py)
slice_daily_as_of(df, as_of_date, include_as_of_date) – Slice a daily DataFrame to rows available as of that date (default: exclude rebalance day = no look-ahead).
get_daily_as_of(symbol, as_of_date, ...) – Load daily data for one symbol as of a date (correct start/end, then slice).
get_daily_batch_as_of(symbols, as_of_date, ...) – Same for multiple symbols.
rebalance_dates(start, end, freq) – Generate rebalance dates (e.g. month-end "ME", quarter-end "QE") for walk-forward/backtests.
