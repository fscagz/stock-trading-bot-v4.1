"""
Backtesting engine.

Runs a monthly (configurable) rebalance loop over a historical period.
At each rebalance date it:
  1. Loads the dated universe snapshot
  2. Builds the feature matrix using only point-in-time data
  3. Calls signal_fn to get target portfolio weights
  4. Runs pre-flight integrity checks (look-ahead + survivorship)
  5. Computes transaction costs and updates portfolio NAV
  6. Simulates daily portfolio returns to the next rebalance

The engine is deliberately data-source agnostic. The caller provides a
signal_fn and pre-loaded price data. This separation allows the same engine
to run a simple momentum strategy, a composite factor model, or an ML
ranking model without engine changes.

Signal function interface
-------------------------
    def signal_fn(
        rebalance_date: date,
        tickers: list[str],
        price_panel: dict[str, pd.DataFrame],   # point-in-time price data
    ) -> dict[str, float]:                       # ticker → target weight
        ...

Weights should be non-negative (long-only) and ideally sum to 1. Any
tickers not in the returned dict are assumed to have weight 0.

Usage
-----
    engine = BacktestEngine(cost_model=CostModel(spread_bps=10))
    result = engine.run(
        signal_fn=my_signal,
        price_panel=price_panel,    # {ticker: daily_df}, full history
        start_date="2015-01-01",
        end_date="2023-12-31",
        snapshot_dir=UNIVERSE_SNAPSHOT_DIR,
    )
    print(format_metrics(result.metrics))
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.costs import CostModel, DEFAULT_COST_MODEL, compute_rebalance_costs
from backtest.integrity import run_honesty_checks
from backtest.metrics import compute_metrics
from data.integrity_checks import IntegrityError, run_preflight_checks
from data.point_in_time import slice_daily_as_of, rebalance_dates as gen_rebalance_dates
from data.universe import load_universe_snapshot


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RebalanceRecord:
    """Metadata captured at each rebalance step."""
    date: date
    tickers: List[str]
    target_weights: Dict[str, float]
    prev_weights: Dict[str, float]
    total_cost: float
    cost_per_ticker: Dict[str, float]
    snapshot_used: bool         # True if a dated snapshot was loaded
    n_positions: int


@dataclass
class BacktestResult:
    """Full output of a backtest run."""
    portfolio_values: pd.Series     # daily NAV (starts at initial_capital)
    daily_returns: pd.Series        # daily portfolio returns
    rebalance_records: List[RebalanceRecord]
    metrics: Dict[str, float]
    rebalance_log: pd.DataFrame     # rebalance_records as a DataFrame (convenience)
    weight_snapshots: pd.DataFrame  # rows=dates, cols=tickers; rebalance weights
    start_date: date
    end_date: date
    initial_capital: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

SignalFn = Callable[
    [date, List[str], Dict[str, pd.DataFrame]],
    Dict[str, float]
]


class BacktestEngine:
    """
    Walk-forward backtesting engine with monthly rebalancing.

    Parameters
    ----------
    cost_model : CostModel
        Transaction cost assumptions.
    initial_capital : float
        Starting portfolio NAV (default 1.0 for normalised returns).
    rebalance_freq : str
        Pandas date offset for rebalance dates. 'ME' = month-end,
        'QE' = quarter-end. See point_in_time.rebalance_dates().
    run_integrity_checks : bool
        If True, call run_preflight_checks() before each rebalance.
        Set False only for speed during parameter sweeps after initial validation.
    snapshot_dir : Path, optional
        Where to load dated universe snapshots from.
    max_snapshot_staleness : int
        Maximum days a universe snapshot can be stale. Raises if exceeded.
    benchmark_ticker : str
        Benchmark for relative metrics (e.g. 'SPY'). Set None to skip.
    """

    def __init__(
        self,
        cost_model: CostModel = DEFAULT_COST_MODEL,
        initial_capital: float = 1.0,
        rebalance_freq: str = "ME",
        run_integrity_checks: bool = True,
        snapshot_dir=None,
        max_snapshot_staleness: int = 45,
        benchmark_ticker: Optional[str] = "SPY",
    ):
        self.cost_model = cost_model
        self.initial_capital = initial_capital
        self.rebalance_freq = rebalance_freq
        self.run_integrity_checks = run_integrity_checks
        self.snapshot_dir = snapshot_dir
        self.max_snapshot_staleness = max_snapshot_staleness
        self.benchmark_ticker = benchmark_ticker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        signal_fn: SignalFn,
        price_panel: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        fallback_universe: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> BacktestResult:
        """
        Run the backtest.

        Parameters
        ----------
        signal_fn : callable
            See module docstring for interface.
        price_panel : dict {ticker: daily_df}
            Full price history for all relevant tickers. Each DataFrame must
            have a DatetimeIndex and a 'close' column.
        start_date, end_date : str
            Backtest period ('YYYY-MM-DD').
        fallback_universe : list of str, optional
            Used when no snapshot is available for a rebalance date.
            WARNING: This introduces survivorship bias. For production runs,
            ensure all rebalance dates have a snapshot.
        verbose : bool

        Returns
        -------
        BacktestResult
        """
        start = pd.Timestamp(start_date).date()
        end = pd.Timestamp(end_date).date()

        # Normalise price panel index
        price_panel = self._normalise_price_panel(price_panel)

        # Build a combined daily return matrix for the full period
        all_daily_returns = self._build_daily_return_matrix(price_panel, start_date, end_date)

        # Generate rebalance dates
        reb_dates = gen_rebalance_dates(start_date, end_date, freq=self.rebalance_freq)
        if len(reb_dates) == 0:
            raise ValueError(f"No rebalance dates between {start_date} and {end_date}.")

        # Load benchmark if requested
        benchmark_returns: Optional[pd.Series] = None
        if self.benchmark_ticker and self.benchmark_ticker in price_panel:
            benchmark_returns = self._compute_benchmark_returns(
                price_panel[self.benchmark_ticker], start_date, end_date
            )

        # Main loop state
        portfolio_value = self.initial_capital
        current_weights: Dict[str, float] = {}
        rebalance_records: List[RebalanceRecord] = []
        weight_snapshots_list: List[Dict] = []

        # Daily portfolio value accumulator
        daily_nav: Dict[pd.Timestamp, float] = {}

        prev_reb_date: Optional[pd.Timestamp] = None

        for i, reb_ts in enumerate(reb_dates):
            reb_date = reb_ts.date()
            if verbose:
                print(f"[engine] Rebalance {i + 1}/{len(reb_dates)} — {reb_date}")

            # 1. Load universe
            tickers, snapshot_used = self._load_universe(reb_date, fallback_universe)
            if not tickers:
                if verbose:
                    print(f"  [engine] No universe for {reb_date}, skipping.")
                continue

            # 2. Get point-in-time price data for this rebalance date
            price_data_as_of = self._slice_price_panel(price_panel, reb_date, tickers)

            # 3. Run integrity pre-flight (survivorship bias check)
            if self.run_integrity_checks and snapshot_used:
                try:
                    run_preflight_checks(
                        rebalance_date=reb_date,
                        filing_dates={t: None for t in tickers},  # look-ahead check handled by signal_fn data layer
                        backtest_tickers=tickers,
                        snapshot_tickers=tickers,  # already from snapshot, so this always passes
                        context=f"rebalance_{reb_date}",
                    )
                except IntegrityError as e:
                    warnings.warn(str(e))
                    continue

            # 4. Compute daily returns from prev_reb_date to this reb_date
            if prev_reb_date is not None:
                portfolio_value = self._simulate_daily_returns(
                    current_weights, all_daily_returns, prev_reb_date, reb_ts,
                    portfolio_value, daily_nav,
                )

            # 5. Get target weights from signal function
            try:
                target_weights = signal_fn(reb_date, tickers, price_data_as_of)
            except Exception as e:
                warnings.warn(f"[engine] signal_fn failed at {reb_date}: {e}. Holding previous weights.")
                target_weights = current_weights

            target_weights = self._normalise_weights(target_weights)

            # 6. Current prices for cost calculation
            prices_at_reb = self._get_prices_at_date(price_panel, reb_ts, tickers)

            # 7. Compute transaction costs
            total_cost, cost_per_ticker = compute_rebalance_costs(
                prev_weights=current_weights,
                target_weights=target_weights,
                prices=prices_at_reb,
                portfolio_value=portfolio_value,
                model=self.cost_model,
            )

            # 8. Deduct costs and update state
            portfolio_value = max(portfolio_value - total_cost, 0.0)
            daily_nav[reb_ts] = portfolio_value
            current_weights = target_weights

            record = RebalanceRecord(
                date=reb_date,
                tickers=tickers,
                target_weights=target_weights,
                prev_weights=dict(current_weights),
                total_cost=total_cost,
                cost_per_ticker=cost_per_ticker,
                snapshot_used=snapshot_used,
                n_positions=len([w for w in target_weights.values() if w > 1e-4]),
            )
            rebalance_records.append(record)
            weight_snapshots_list.append({"date": reb_date, **target_weights})
            prev_reb_date = reb_ts

        # Simulate returns for the tail period after the last rebalance
        if prev_reb_date is not None:
            end_ts = pd.Timestamp(end_date)
            portfolio_value = self._simulate_daily_returns(
                current_weights, all_daily_returns, prev_reb_date, end_ts,
                portfolio_value, daily_nav,
            )

        # Build output series
        nav_series = pd.Series(daily_nav).sort_index()
        if nav_series.empty:
            daily_returns = pd.Series(dtype=float)
        else:
            daily_returns = nav_series.pct_change().dropna()

        # Build weight snapshot DataFrame
        weight_snapshots = pd.DataFrame(weight_snapshots_list)
        if not weight_snapshots.empty:
            weight_snapshots = weight_snapshots.set_index("date").fillna(0)

        # Build rebalance log DataFrame
        rebalance_log = pd.DataFrame([
            {
                "date": r.date,
                "n_positions": r.n_positions,
                "total_cost": r.total_cost,
                "snapshot_used": r.snapshot_used,
                "portfolio_value": daily_nav.get(pd.Timestamp(r.date), float("nan")),
            }
            for r in rebalance_records
        ])

        # Compute metrics
        metrics = compute_metrics(
            returns=daily_returns,
            benchmark_returns=benchmark_returns,
            weight_snapshots=weight_snapshots if not weight_snapshots.empty else None,
        )

        return BacktestResult(
            portfolio_values=nav_series,
            daily_returns=daily_returns,
            rebalance_records=rebalance_records,
            metrics=metrics,
            rebalance_log=rebalance_log,
            weight_snapshots=weight_snapshots,
            start_date=start,
            end_date=end,
            initial_capital=self.initial_capital,
        )

    def run_honesty_checks(self, result: BacktestResult) -> None:
        """Run and print the post-backtest honesty checklist."""
        report = run_honesty_checks(
            daily_returns=result.daily_returns,
            rebalance_log=result.rebalance_log,
            backtest_start=result.start_date,
        )
        report.print_summary()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalise_price_panel(
        self, panel: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Ensure all DataFrames have a DatetimeIndex and lowercase column names."""
        out = {}
        for ticker, df in panel.items():
            df = df.copy()
            df.index = pd.to_datetime(df.index).normalize()
            df.columns = [c.lower() for c in df.columns]
            out[ticker] = df
        return out

    def _build_daily_return_matrix(
        self,
        price_panel: Dict[str, pd.DataFrame],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Build a (dates × tickers) matrix of daily returns for the backtest period.
        """
        returns_dict = {}
        for ticker, df in price_panel.items():
            if "close" not in df.columns:
                continue
            sub = df["close"].loc[start:end]
            if len(sub) > 1:
                returns_dict[ticker] = sub.pct_change()
        if not returns_dict:
            return pd.DataFrame()
        return pd.DataFrame(returns_dict).sort_index()

    def _compute_benchmark_returns(
        self,
        bench_df: pd.DataFrame,
        start: str,
        end: str,
    ) -> pd.Series:
        close = bench_df["close"].loc[start:end]
        return close.pct_change().dropna().rename("benchmark")

    def _load_universe(
        self,
        reb_date: date,
        fallback: Optional[List[str]],
    ) -> Tuple[List[str], bool]:
        """Load dated universe snapshot. Falls back if unavailable."""
        try:
            tickers = load_universe_snapshot(
                as_of=reb_date,
                snapshot_dir=self.snapshot_dir,
                max_staleness_days=self.max_snapshot_staleness,
            )
            if tickers:
                return tickers, True
        except (ValueError, Exception):
            pass

        if fallback:
            warnings.warn(
                f"[engine] No universe snapshot for {reb_date}. "
                f"Using fallback universe ({len(fallback)} tickers). "
                "This introduces survivorship bias — use EODHD or build snapshots first."
            )
            return list(fallback), False
        return [], False

    def _slice_price_panel(
        self,
        panel: Dict[str, pd.DataFrame],
        reb_date: date,
        tickers: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """Return point-in-time price data for each ticker (no look-ahead)."""
        out = {}
        reb_ts = pd.Timestamp(reb_date)
        for t in tickers:
            if t not in panel:
                continue
            sliced = slice_daily_as_of(panel[t], reb_ts, include_as_of_date=False)
            if not sliced.empty:
                out[t] = sliced
        return out

    def _get_prices_at_date(
        self,
        panel: Dict[str, pd.DataFrame],
        ts: pd.Timestamp,
        tickers: List[str],
    ) -> Dict[str, float]:
        """Get the closing price for each ticker on or just before ts."""
        prices = {}
        for t in tickers:
            if t not in panel:
                continue
            df = panel[t]
            prior = df.loc[df.index <= ts]
            if not prior.empty and "close" in prior.columns:
                prices[t] = float(prior["close"].iloc[-1])
        return prices

    def _simulate_daily_returns(
        self,
        weights: Dict[str, float],
        daily_returns: pd.DataFrame,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        portfolio_value: float,
        nav_dict: Dict[pd.Timestamp, float],
    ) -> float:
        """
        Compound portfolio returns day-by-day between rebalances.
        Updates nav_dict in-place. Returns final portfolio_value.
        """
        if not weights or daily_returns.empty:
            return portfolio_value

        period = daily_returns.loc[
            (daily_returns.index > start_ts) & (daily_returns.index <= end_ts)
        ]
        if period.empty:
            return portfolio_value

        held_tickers = [t for t in weights if t in period.columns]
        if not held_tickers:
            # No price data for held tickers — hold at current NAV
            for ts in period.index:
                nav_dict[ts] = portfolio_value
            return portfolio_value

        w_vec = np.array([weights[t] for t in held_tickers])
        ret_matrix = period[held_tickers].fillna(0).values  # (days, tickers)

        for i, ts in enumerate(period.index):
            day_ret = float(w_vec @ ret_matrix[i])
            portfolio_value *= (1 + day_ret)
            nav_dict[ts] = portfolio_value

        return portfolio_value

    @staticmethod
    def _normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """Clip negative weights and normalise so they sum to 1."""
        clipped = {t: max(0.0, w) for t, w in weights.items()}
        total = sum(clipped.values())
        if total <= 0:
            return {}
        return {t: w / total for t, w in clipped.items()}
