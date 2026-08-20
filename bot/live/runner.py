"""
V4 live runner — short and long momentum strategies with independent heat budgets.

On each 1-min bar:
  1. Update ATR / VWAP indicators
  2. If position open: check stop/target/manager exits → cancel broker stop → cover/sell
  3. If no position, from market open (9:30): run Stage-2 validator → enter → submit broker stop
  4. Force-close all remaining positions at 15:55 ET

Threading model
---------------
- BarStream.run_with_reconnect() runs in the main thread (blocking asyncio
  event loop). It automatically reconnects on websocket drop.
- _on_bar() is called from the event loop thread — all portfolio and state
  mutations happen here, no lock needed for those structures.
- The watchlist refresh and status log run as daemon threads. They only call
  BarStream.subscribe()/unsubscribe() (queue-based, thread-safe) and read
  _open_syms under _open_syms_lock.

Heat budgets
------------
Short and long strategies maintain separate PortfolioState instances so their
heat limits are fully independent. A full short book never blocks a long entry
and vice versa. Both portfolios share the same equity value, which is synced
after every PnL event via _sync_equity().

Broker-side stops
-----------------
After every entry a protective stop is submitted to Alpaca at the stop price:
a stop-SELL for longs, a stop-BUY for shorts. If the process dies, Alpaca's stop
closes the position automatically. When our code exits a position for any reason,
it cancels the broker stop first. If the cancel fails because the stop already
filled (broker closed while we were monitoring), the manual exit order is skipped
— preventing a double order. When the trailing stop moves, the old broker stop is
cancelled and a new one submitted at the updated price.

Restart reconciliation
----------------------
On startup, session_state.json (today's date only) is cross-referenced with
Alpaca's actual open positions. Positions found in Alpaca but not our state get
reconstructed with conservative stops. Positions in our state but NOT in Alpaca
(broker stop fired while offline) are cleaned up. A fresh broker stop is
re-submitted for every reconciled position.

Run with:  python -m bot.live [--risk-scale 0.5]
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import copy

import bot.broker_alpaca as broker
import bot.broker_ibkr as broker_ibkr
from bot.backtest.news_filter import NewsFilter
from bot.config import V4Config, make_gap_hold_config
from bot.dashboard.state import DashboardState
from bot.data.regime import RegimeFilter
from bot.intraday.data.stream import BarStream
from bot.intraday.indicators.atr import ATRIndicator
from bot.intraday.indicators.vwap import VWAPIndicator
from bot.intraday.risk.portfolio import PortfolioState
from bot.intraday.risk.sizing import compute_position_size
from bot.intraday.types import Bar, Position, TradeRecord
from bot.live.state import SessionState
from bot.intraday.risk.kill_switch import KillSwitch
from bot.momentum.validator import MomentumValidator
from bot.positions.manager import PositionManager
from bot.trade_logger import TradeLogger
from bot.short_qual_logger import ShortQualLogger

_ET = ZoneInfo("America/New_York")
_ENTRY_START = dtime(9, 30)  # entries open from market open
_EOD_FORCE_CLOSE = dtime(15, 55)
_PULLBACK_LIMIT_CUTOFF = dtime(15, 45)  # don't arm new pullback limits after this time
_WATCHLIST_REFRESH_SEC = 30
_STATUS_LOG_SEC = 60
_ETB_REFRESH_SEC = 1800        # refresh ETB list every 30 minutes

# Hybrid entry parameters — H+ finalist configuration.
# Tier-4 signals (confidence_multiplier ≥ 4.0) enter at market with 4×ATR target.
# Tier 1–3 signals arm a pullback limit 2×ATR below the signal close, valid for 60 min,
# filling with a 2.5×ATR target and the same 1.5×ATR stop.
# On narrow days (<35 symbols active), additionally require close ≥ 10-day daily MA.
_HYBRID_PULLBACK_ATR        = 2.0   # limit depth: N×ATR below signal close
_HYBRID_PULLBACK_TTL_MIN    = 60    # cancel unfilled pullback limit after N minutes
_HYBRID_PULLBACK_TARGET_ATR = 2.5   # target multiplier for pullback fills
_HYBRID_CHASE_TIER          = 4.0   # confidence_multiplier threshold for immediate market entry
_HYBRID_NARROW_BREADTH      = 35    # fewer active symbols than this = narrow day
_HYBRID_MA_PERIOD           = 10    # narrow-day filter: require close ≥ N-day daily MA

# Gap-and-hold entry: stock gaps ≥5% at open vs prior close, holds within 2% of the
# day open for 15 bars (30s bars → 7.5 min), then enters at market.
_GAP_HOLD_MIN_PCT    = 0.05   # minimum gap at open vs prior close
_GAP_HOLD_BARS       = 15     # bars price must hold near the day open — see LiveRunner.__init__ for runtime value
_GAP_HOLD_TOLERANCE  = 0.02   # max retracement from day open before gap is "broken"

# Change #1 (2026-06-30): momentum-at-entry filter. "Held the gap" proves inertia,
# not momentum — over 2 weeks, 58 hard-stops (the entire net loss) came from flat
# stocks that ticked DOWN on the entry bar. Require the entry bar to close above the
# prior bar so we only buy when price is actually rising at entry.
_GAP_ENTRY_REQUIRE_RISING = True

# Change #3 (2026-06-30): circuit breaker. The bot took 54 trades on 2026-06-24
# (death by a thousand cuts). Halt new entries for the session after a losing streak
# or a daily trade-count cap. 4 consec losses preserves the bounce-back winner
# (e.g. RZLV +$717 came after 3 straight losses on 2026-06-30) while stopping a rout.
_CB_MAX_CONSEC_LOSSES  = 4    # halt new entries after N consecutive losing closes
_CB_MAX_TRADES_PER_DAY = 10   # halt new entries after N total entries in a session

# HOD-rejection short entry: fade a parabolic after the day's high is rejected.
# Stock must have run ≥N% from its day open; wait for 10 consecutive bars without
# a new intraday high; stop above the day's high, target below entry.
# 2026-06-30 SHORTS INVESTIGATION: lowered 50.0 → 25.0. At 50% from day-open almost
# nothing qualified intraday → zero shorts fired in two weeks. The dominant pattern in
# our long losses is gap-up-then-FADE, which is exactly what this fades. 25% lets the
# pattern actually qualify on real movers; protective stop (HOD + 2×ATR) is unchanged.
_SHORT_HOD_MIN_RUN_PCT = 25.0   # min gain from day open before the pattern qualifies
_SHORT_HOD_REJ_BARS    = 10     # bars without new intraday high before entry fires — see LiveRunner.__init__ for runtime value
_SHORT_HOD_STOP_MULT   = 2.0    # stop: HOD + N×ATR (above the day's peak, not entry)
_SHORT_HOD_TARGET_MULT = 2.0    # target: entry − N×ATR

logger = logging.getLogger(__name__)


class LiveRunner:
    """Subscribes to Alpaca real-time 1-min bars and executes the V4 short and long strategies.

    Short and long positions are tracked in separate PortfolioState instances so their
    heat budgets are fully independent — a full short book never blocks a long entry.
    """

    def __init__(
        self,
        ibkr_host: str,
        ibkr_port: int,
        ibkr_client_id: int,
        ibkr_scanner_client_id: int,
        api_key: str,
        secret_key: str,
        short_config: V4Config,
        long_config: V4Config,
        equity: float,
        etb_set: Set[str],
        risk_scale: float = 1.0,
        dash: Optional[DashboardState] = None,
        trade_log_dir: str = "logs",
        enable_shorts: bool = False,
    ) -> None:
        if risk_scale != 1.0:
            for cfg in (short_config, long_config):
                cfg.risk_per_trade = round(cfg.risk_per_trade * risk_scale, 6)
                # max_position_pct is a hard concentration cap, not a risk parameter — don't scale it
                cfg.max_portfolio_heat = min(round(cfg.max_portfolio_heat * risk_scale, 4), 1.0)
            logger.info(
                "Risk scale %.2f×: short risk_per_trade=%.4f heat=%.4f | "
                "long risk_per_trade=%.4f heat=%.4f",
                risk_scale,
                short_config.risk_per_trade, short_config.max_portfolio_heat,
                long_config.risk_per_trade, long_config.max_portfolio_heat,
            )

        self._ibkr_host = ibkr_host
        self._ibkr_port = ibkr_port
        self._ibkr_client_id = ibkr_client_id
        self._ibkr_scanner_client_id = ibkr_scanner_client_id
        self._short_cfg = short_config
        self._long_cfg = long_config
        self._etb_set = etb_set
        self._shorts_enabled = enable_shorts
        self._api_key = api_key
        self._secret_key = secret_key

        self._news_filter = NewsFilter(api_key, secret_key, cache_only=False)
        self._regime_filter = RegimeFilter()                    # SPY vs 20-day MA (long gate)
        self._short_regime_filter = RegimeFilter(ma_period=50)  # SPY vs 50-day MA (short gate)
        self._regime: dict = {"date": None, "uptrend": True, "short_allowed": True}  # re-evaluated each trading day
        self._dash = dash
        self._trade_logger = TradeLogger(log_dir=trade_log_dir)
        self._short_qual_logger = ShortQualLogger(log_dir=trade_log_dir)
        self._open_records: Dict[str, TradeRecord] = {}

        # --- Independent heat budgets (event loop thread owned) ---
        # Both portfolios share the same equity value, synced via _sync_equity().
        self._short_portfolio = PortfolioState(equity=equity, config=short_config)
        self._long_portfolio = PortfolioState(equity=equity, config=long_config)

        self._short_validator = MomentumValidator(short_config)
        self._long_validator = MomentumValidator(long_config)
        self._short_manager = PositionManager(short_config)
        self._long_manager = PositionManager(long_config)
        self._long_kill_switch = KillSwitch(long_config)
        self._atrs: Dict[str, ATRIndicator] = {}
        self._vwaps: Dict[str, VWAPIndicator] = {}
        self._entered_today: Set[str] = set()
        self._trading_date: Optional[date] = None   # used to detect day rollover
        self._day_highs: Dict[str, float] = {}      # running intraday high per symbol (for dist-from-high filter)
        self._subscribed_today: Set[str] = set()    # symbols added to watchlist today — never removed mid-session
        # Hybrid entry state
        self._armed_limits: Dict[str, dict] = {}    # sym → {order_id, limit, atr, shares, mult, armed_at}
        self._daily_ma10: Dict[str, float] = {}     # sym → 10-day daily close MA (for narrow-day filter)
        self._prev_bars: Dict[str, Bar] = {}        # sym → previous bar (for vol-vs-prev-bar filter)

        # Bar-count thresholds — derived from bar_size_seconds so all time windows stay
        # consistent regardless of bar resolution (15s, 30s, 60s).
        _bar_secs = long_config.bar_size_seconds
        self._gap_hold_bars = max(1, round(450 / _bar_secs))    # 7.5 min window
        self._short_hod_rej_bars = max(1, round(300 / _bar_secs))  # 5 min window
        logger.info(
            "Bar resolution: %ds → gap_hold_bars=%d (%.1f min), hod_rej_bars=%d (%.1f min)",
            _bar_secs, self._gap_hold_bars, self._gap_hold_bars * _bar_secs / 60,
            self._short_hod_rej_bars, self._short_hod_rej_bars * _bar_secs / 60,
        )

        # Gap-and-hold entry state (reset each trading day)
        self._day_opens: Dict[str, float] = {}      # sym → first bar's open price
        self._prev_closes: Dict[str, float] = {}    # sym → prior day's close (persists across days)
        self._gap_tracking: Dict[str, int] = {}     # sym → bar count since gap detected
        self._gap_confirmed: Set[str] = set()       # syms where gap held ≥ self._gap_hold_bars
        self._gap_rejection_logged: Set[str] = set()  # syms where catalyst rejection was already logged today
        self._gap_hold_losses: Dict[str, int] = {}  # sym → same-day gap-hold loss count; blocks re-entry
        self._gap_hold_entry_count: int = 0          # total gap-hold entries today (vs max_gap_hold_entries_per_day)
        self._entries_today: int = 0                 # all entries today (gap-hold + long + short) — circuit breaker
        self._trading_halted_today: bool = False     # circuit breaker latched for the session

        # HOD-rejection short entry state (reset each trading day)
        self._short_hod: Dict[str, dict] = {}       # sym → {hod, bars_since_hod, qualified}

        # News catalysts are pre-fetched in the watchlist daemon thread so the
        # event-loop thread never blocks on an HTTP call inside _on_bar.
        self._news_warmed: Set[str] = set()
        self._news_warmed_date: Optional[date] = None

        # --- Shared (see threading model above) ---
        self._baseline_vols: Dict[str, float] = {}
        self._baseline_vols_lock = threading.Lock()
        self._last_prices: Dict[str, float] = {}
        self._open_syms: Set[str] = set()
        self._open_syms_lock = threading.Lock()

        # --- Session state (persisted to disk) ---
        self._state = SessionState()

        # --- IBKR bar cache (written to backtest cache at EOD) ---
        self._live_bar_cache: Dict[str, List[dict]] = {}
        self._bar_cache_dir = Path("ibkr_bars")
        self._bar_cache_dir.mkdir(parents=True, exist_ok=True)

        self._stream: Optional[BarStream] = None
        self._stop_event = threading.Event()
        self._current_watchlist: Set[str] = set()

        # Protects the claim-and-remove step in _execute_close and
        # _submit_eod_close_orders so a bar event and the EOD timer thread
        # can't both close the same position and submit duplicate broker orders.
        self._close_lock = threading.Lock()

    def _sync_equity(self, new_equity: float) -> None:
        """Propagate an equity change to both portfolio states."""
        self._short_portfolio.equity = new_equity
        self._long_portfolio.equity = new_equity

    def _reset_daily_state(self) -> None:
        """Reset all per-session state at the start of each new trading day.

        Called on the first bar of a new session so the bot behaves correctly
        when running 24/7 without restarts.
        """
        # Cancel any unfilled pullback limit orders that survived from yesterday
        for sym, armed in list(self._armed_limits.items()):
            try:
                broker.cancel_order(armed["order_id"])
                logger.info("Cancelled stale pullback limit for %s (order %s)", sym, armed["order_id"])
            except Exception:
                pass
        self._armed_limits.clear()
        self._short_hod.clear()

        # Reconcile equity with broker before resetting session_start_equity
        # so daily_pnl_pct() is accurate for the new day.
        try:
            actual_equity = broker.get_account_info()["portfolio_value"]
            self._sync_equity(actual_equity)
            logger.info("Daily equity sync: $%.2f", actual_equity)
        except Exception as exc:
            logger.warning("Could not sync equity at day rollover: %s — using last known value", exc)

        self._entries_today = 0
        self._trading_halted_today = False

        for portfolio in (self._long_portfolio, self._short_portfolio):
            portfolio.session_start_equity  = portfolio.equity
            portfolio.kill_switch_active    = False
            portfolio.consecutive_losses    = 0
            portfolio.cooldown_until        = None

        # 2026-07-01 bug fix: bulk baseline volume/prev_close load only ran once,
        # at process startup (see run()). A process that spans midnight (e.g. restarted
        # post-close the night before) never re-fetches movers for the new day — new
        # symbols only trickle in one-at-a-time via the unreliable incremental fetch in
        # the watchlist-refresh loop, which mostly fails for single-symbol requests.
        # Result: today's actual movers never get prev_close data, so gap-hold can never
        # confirm even with 100+ watched symbols. Re-run the same bulk fetch run() does,
        # using a fresh mover list, so a new trading day always starts with real coverage.
        # Runs on a background thread — _reset_daily_state() is called synchronously from
        # _on_bar() on the event-loop thread, which must never block on network calls
        # (same rule the watchlist-refresh loop and news pre-fetch already follow).
        threading.Thread(
            target=self._refresh_movers_and_baselines_for_new_day,
            daemon=True,
        ).start()

    def _refresh_movers_and_baselines_for_new_day(self) -> None:
        try:
            movers = broker_ibkr.get_movers(
                self._ibkr_host, self._ibkr_port,
                self._ibkr_scanner_client_id, top_n=50,
            )
            fresh_watchlist = self._build_watchlist_from_movers(movers)
            if fresh_watchlist:
                self.load_baseline_volumes(list(fresh_watchlist))
                # Atomic pointer swap rather than in-place |= — safe for lock-free
                # cross-thread reads elsewhere (matches _current_watchlist's existing
                # unsynchronized access pattern in the watchlist-refresh loop).
                self._current_watchlist = self._current_watchlist | fresh_watchlist
                logger.info(
                    "Day-rollover baseline refresh: %d movers re-fetched", len(fresh_watchlist),
                )
        except Exception as exc:
            logger.warning("Day-rollover baseline refresh failed: %s — relying on incremental fetch", exc)

        # Refresh baseline volumes for all currently-watched symbols so the
        # 20-day average stays current rather than using stale day-1 values.
        with self._baseline_vols_lock:
            watched = list(self._baseline_vols.keys())
        if watched:
            self._fetch_baseline_volumes(watched)

        logger.info(
            "Daily state reset complete — kill switch cleared, "
            "session equity anchor updated, %d baseline volumes refreshed",
            len(watched),
        )

    def _sync_dash(self, closed_record: Optional[TradeRecord] = None) -> None:
        if self._dash is None:
            return
        with self._dash._lock:
            self._dash.equity = self._equity
            # Approximate cash as equity minus capital committed to open long positions.
            # Short positions add cash (proceeds), so we subtract their notional from invested.
            long_invested = sum(
                p.entry_price * p.shares for p in self._long_portfolio.positions.values()
            )
            short_proceeds = sum(
                p.entry_price * p.shares for p in self._short_portfolio.positions.values()
            )
            self._dash.cash = max(0.0, self._equity - long_invested + short_proceeds)
            self._dash.positions = {
                **self._short_portfolio.positions,
                **self._long_portfolio.positions,
            }
            lp = self._long_portfolio
            self._dash.portfolio_heat_pct = lp.portfolio_heat_pct
            self._dash.kill_switch_active = lp.kill_switch_active
            self._dash.consecutive_losses = lp.consecutive_losses
            self._dash.cooldown_until = lp.cooldown_until
            self._dash.short_heat_pct = self._short_portfolio.portfolio_heat_pct
            reg = self._regime  # single read — daemon may swap the dict concurrently
            self._dash.regime_uptrend = reg["uptrend"]
            self._dash.regime_date = reg["date"]
            self._dash.short_allowed = reg.get("short_allowed", True)
            if closed_record is not None:
                self._dash.closed_trades.append(closed_record)

    @property
    def _equity(self) -> float:
        return self._short_portfolio.equity

    # ------------------------------------------------------------------
    # Indicator helpers (event loop thread only)
    # ------------------------------------------------------------------

    def _get_or_create_indicators(self, sym: str):
        if sym not in self._atrs:
            # Aggregate to 1-minute buckets: the stream delivers sub-minute bars
            # (bar_size_seconds=15/30) and ATR(14) on those measures seconds of
            # range — stops sized off it sit inside ordinary noise. Backtests
            # compute ATR on 1-min Alpaca bars, so this also keeps live stop
            # distances comparable to backtested ones.
            self._atrs[sym] = ATRIndicator(period=14, aggregate_seconds=60)
            self._vwaps[sym] = VWAPIndicator()
        return self._atrs[sym], self._vwaps[sym]

    # ------------------------------------------------------------------
    # Broker stop management (event loop thread only)
    # ------------------------------------------------------------------

    def _submit_broker_stop(self, sym: str, position: Position) -> None:
        """Submit a broker-side protective stop and store the order ID.

        Longs get a stop-SELL (sells if price falls to the stop); shorts get a
        stop-BUY (covers if price rises to the stop). Submitting the wrong side
        leaves the position unprotected at the broker and, if triggered, would
        *add* to the position instead of closing it.
        """
        try:
            if position.direction == "short":
                order_id = broker.submit_stop_buy_order(sym, position.shares, position.stop_price)
            else:
                order_id = broker.submit_stop_order(sym, position.shares, position.stop_price)
            position.stop_order_id = order_id
            logger.debug(
                "Broker %s stop submitted for %s @ %.2f (order %s)",
                position.direction, sym, position.stop_price, order_id,
            )
        except Exception as exc:
            logger.error(
                "Failed to submit broker stop for %s @ %.2f: %s — position unprotected at broker level",
                sym, position.stop_price, exc,
            )

    def _cancel_broker_stop(self, sym: str, position: Position) -> bool:
        """Cancel the broker stop order before we cover manually.

        Returns True  → we must still submit the cover order.
        Returns False → broker already filled the stop (covered for us), skip cover.
        """
        order_id = position.stop_order_id
        if not order_id:
            return True

        try:
            broker.cancel_order(order_id)
            return True  # Cancelled cleanly — we need to cover
        except Exception as exc:
            msg = str(exc).lower()
            if any(word in msg for word in ("filled", "already", "not found", "404", "cannot")):
                # Stop order was already executed by Alpaca — don't double-buy
                logger.info(
                    "Broker stop for %s already filled (order %s) — skipping cover order",
                    sym, order_id,
                )
                return False
            # Unknown error — err on the side of covering to avoid uncovered position
            logger.warning(
                "Could not cancel broker stop for %s (order %s): %s — covering anyway",
                sym, order_id, exc,
            )
            return True

    def _replace_broker_stop(self, sym: str, position: Position, portfolio: PortfolioState) -> None:
        """Cancel old broker stop and submit a new one at the current stop price.

        If the old stop already filled at the broker (trailing logic runs off
        bar closes, so the broker can fill on an intra-bar tick before our own
        stop_hit check on the next bar), the position is already gone at the
        broker. Submitting a fresh sell-stop then reads to Alpaca as opening a
        short (rejected if not ETB), while our own book still shows the
        position open — a phantom position that desyncs equity and blocks EOD
        close. Route through _execute_close instead so the position is closed
        on our side using the same accounting as any other exit.
        """
        old_id = position.stop_order_id
        if old_id:
            try:
                broker.cancel_order(old_id)
            except Exception as exc:
                msg = str(exc).lower()
                if any(word in msg for word in ("filled", "already", "not found", "404", "cannot")):
                    logger.info(
                        "Broker stop for %s already filled (order %s) during trail-update — closing internally",
                        sym, old_id,
                    )
                    self._execute_close(sym, position.stop_price, "hard_stop", portfolio)
                    return
                logger.warning("Could not cancel old broker stop %s for %s: %s", old_id, sym, exc)
        self._submit_broker_stop(sym, position)
        self._state.update_stop(sym, position.stop_price, position.stop_order_id)

    # ------------------------------------------------------------------
    # Bar handler — runs in the asyncio event loop thread
    # ------------------------------------------------------------------

    def _on_bar(self, bar: Bar) -> None:
        sym = bar.symbol
        bar_et = bar.timestamp.astimezone(_ET)
        bar_time = bar_et.time()
        today = bar_et.date()

        # Reset per-day state on the first bar of a new trading session
        if self._trading_date != today:
            if self._trading_date is not None:
                self._flush_bar_cache()   # write previous day's bars before clearing
                self._reset_daily_state()
            self._trading_date = today
            self._entered_today.clear()
            self._day_highs.clear()
            self._subscribed_today.clear()
            self._day_opens.clear()
            self._gap_tracking.clear()
            self._gap_confirmed.clear()
            self._gap_rejection_logged.clear()
            self._gap_hold_losses.clear()
            self._gap_hold_entry_count = 0
            logger.info("New trading day %s — daily state reset", today)
            # Restore same-day protection state after a mid-day restart.
            # get_gap_losses() returns empty if the file is from a previous day,
            # so this is a no-op on genuine day rollovers.
            _saved_losses, _saved_entered = self._state.get_gap_losses()
            if _saved_losses:
                self._gap_hold_losses.update(_saved_losses)
                logger.info("Restored gap-hold losses after restart: %s", dict(self._gap_hold_losses))
            if _saved_entered:
                self._entered_today.update(_saved_entered)
                logger.info("Restored entered_today after restart: %d symbols", len(self._entered_today))

        # Track running intraday high for each symbol (for stage2_min_dist_from_day_high_pct)
        if bar.high > self._day_highs.get(sym, 0.0):
            self._day_highs[sym] = bar.high

        # Gap-and-hold tracking: detect gap on first bar, count hold bars toward confirmation
        if sym not in self._day_opens:
            self._day_opens[sym] = bar.open
            pc = self._prev_closes.get(sym, 0.0)
            if pc > 0 and bar.open >= pc * (1.0 + _GAP_HOLD_MIN_PCT):
                self._gap_tracking[sym] = 1
        elif sym in self._gap_tracking and sym not in self._gap_confirmed:
            day_open = self._day_opens[sym]
            if day_open > 0 and bar.close < day_open * (1.0 - _GAP_HOLD_TOLERANCE):
                del self._gap_tracking[sym]  # gap broken — stop counting
            else:
                self._gap_tracking[sym] += 1
                if self._gap_tracking[sym] >= self._gap_hold_bars:
                    self._gap_confirmed.add(sym)
                    logger.info("GAP-HOLD confirmed for %s (gap open=%.2f prev_close=%.2f)",
                                sym, day_open, self._prev_closes.get(sym, 0.0))

        # Accumulate bar for EOD backtest cache write (market hours only)
        if dtime(9, 30) <= bar_et.time() <= dtime(16, 0):
            self._live_bar_cache.setdefault(sym, []).append({
                "t": bar.timestamp.isoformat(),
                "o": bar.open, "h": bar.high, "l": bar.low,
                "c": bar.close, "v": bar.volume,
            })

        atr_ind, vwap_ind = self._get_or_create_indicators(sym)
        atr_val = atr_ind.update(bar)
        vwap_val = vwap_ind.update(bar)
        prev_bar = self._prev_bars.get(sym)   # capture BEFORE overwrite (vol-vs-prev-bar filter)
        self._prev_bars[sym] = bar
        self._last_prices[sym] = bar.close
        if self._dash is not None:
            with self._dash._lock:
                self._dash.last_prices[sym] = bar.close
        with self._baseline_vols_lock:
            baseline = self._baseline_vols.get(sym, 0.0)

        # Update per-symbol HOD state for short entry tracking (market hours only)
        if dtime(9, 30) <= bar_time < _EOD_FORCE_CLOSE:
            if sym not in self._short_hod:
                self._short_hod[sym] = {
                    "hod": bar.high,
                    "bars_since_hod": 0,
                    "qualified": False,
                }
            else:
                state = self._short_hod[sym]
                if bar.high > state["hod"]:
                    state["hod"] = bar.high
                    state["bars_since_hod"] = 0
                else:
                    state["bars_since_hod"] += 1
                if not state["qualified"]:
                    open_ref = self._day_opens.get(sym, 0.0)
                    if open_ref > 0 and (state["hod"] - open_ref) / open_ref * 100 >= _SHORT_HOD_MIN_RUN_PCT:
                        state["qualified"] = True
                        run_pct = (state["hod"] - open_ref) / open_ref * 100
                        in_etb = sym in self._etb_set
                        logger.info(
                            "HOD-REJECTION qualified %s: run=%.1f%% (open=%.2f hod=%.2f) etb=%s",
                            sym, run_pct, open_ref, state["hod"], in_etb,
                        )
                        # 2026-07-01: persist qualification+ETB events to a dedicated CSV
                        # (separate from the scrolling bot.log) so the ETB hit-rate can be
                        # measured over weeks without re-parsing ever-growing log text.
                        self._short_qual_logger.log(
                            ticker=sym, qualified_at=bar.timestamp, run_pct=run_pct,
                            day_open=open_ref, hod_price=state["hod"], etb_at_qualification=in_etb,
                        )

        # Determine which portfolio owns this symbol (if any)
        active_portfolio = None
        if sym in self._short_portfolio.positions:
            active_portfolio = self._short_portfolio
        elif sym in self._long_portfolio.positions:
            active_portfolio = self._long_portfolio

        # Force-close all positions at 15:55 ET; also resolve any armed pullback limits.
        if bar_time >= _EOD_FORCE_CLOSE:
            if active_portfolio is not None:
                logger.info("EOD force-close %s", sym)
                self._execute_close(sym, bar.close, "eod_force", active_portfolio)
            if sym in self._armed_limits:
                # Cancels the resting order; if it already filled in a bar gap, the
                # position is registered (and the EOD timer will then close it) rather
                # than orphaned. bar_time >= cutoff guarantees the cancel branch.
                self._process_armed_limit(sym, bar, bar_time)
            return

        # --- Exit logic for open positions ---
        if active_portfolio is not None:
            position = active_portfolio.positions[sym]
            is_short = position.direction == "short"
            manager = self._short_manager if is_short else self._long_manager
            stop_hit = bar.high >= position.stop_price if is_short else bar.low <= position.stop_price
            target_hit = bar.low <= position.target_price if is_short else bar.high >= position.target_price

            if stop_hit:
                self._execute_close(sym, position.stop_price, "hard_stop", active_portfolio)
            elif target_hit:
                self._execute_close(sym, position.target_price, "target", active_portfolio)
            elif vwap_val is not None and baseline > 0:
                prev_stop = position.stop_price
                instruction = manager.on_bar(bar, position, vwap_val, baseline)
                if instruction:
                    fill = (position.stop_price
                            if instruction.reason in ("hard_stop", "trailing_stop")
                            else (instruction.limit_price if instruction.limit_price else bar.close))
                    self._execute_close(sym, fill, instruction.reason, active_portfolio)
                else:
                    if position.stop_price != prev_stop:
                        logger.info(
                            "Trailing stop updated for %s: %.2f → %.2f",
                            sym, prev_stop, position.stop_price,
                        )
                        self._replace_broker_stop(sym, position, active_portfolio)
                    else:
                        self._state.update_stop(sym, position.stop_price, position.stop_order_id)
            return

        # --- Entry logic ---
        if bar_time < _ENTRY_START:
            return

        # Pullback limit: confirm fill / TTL / EOD cancellation against the broker.
        # Must run before the entered_today gate — sym is already in entered_today
        # once a limit is armed.
        if sym in self._armed_limits:
            self._process_armed_limit(sym, bar, bar_time)
            return

        if sym in self._entered_today:
            return
        if not (atr_val and baseline > 0):
            return

        # Change #3: circuit breaker. Latch a session-wide halt after a losing streak
        # or a daily trade-count cap, so a bad day can't spiral into 50 trades.
        if not self._trading_halted_today:
            consec = self._long_portfolio.consecutive_losses
            if consec >= _CB_MAX_CONSEC_LOSSES:
                self._trading_halted_today = True
                logger.warning(
                    "CIRCUIT BREAKER: %d consecutive losses — halting new entries for the session",
                    consec,
                )
            elif self._entries_today >= _CB_MAX_TRADES_PER_DAY:
                self._trading_halted_today = True
                logger.warning(
                    "CIRCUIT BREAKER: %d entries today (cap %d) — halting new entries for the session",
                    self._entries_today, _CB_MAX_TRADES_PER_DAY,
                )
        if self._trading_halted_today:
            return

        # Check kill switch / cooldown for long portfolio on every entry-eligible bar
        triggered, ks_reason = self._long_kill_switch.check(self._long_portfolio, bar.timestamp)
        if triggered:
            logger.warning("KILL SWITCH: %s — long entries halted for session", ks_reason)
            # Halt resting pullback orders too, or they could still fill after the halt.
            self._cancel_all_armed_limit_orders("kill_switch")
            self._sync_dash()
            return
        if self._long_portfolio.in_cooldown(bar.timestamp):
            return

        # --- Gap-hold entry (long only; news bypass at ≥12% gap, no momentum validator) ---
        # 2026-06-24: removed-filter caused 18 trades/38.9% WR at -$208 avg; reverted to gap threshold filter.
        # 2026-06-24 ~12pm: raised 10%→12% — 10 new trades at -$215 avg, 5/10 hard_stops; CUPR lost 3x today
        # 2026-06-25: daily entry cap (max_gap_hold_entries_per_day) prevents sequential churn when
        #             many gaps confirm simultaneously and rapid hard_stops free up the position cap.
        # ≥12% gap: bypass news (large gaps are self-evident catalysts)
        # <12% gap: require has_catalyst() (small gaps need quality confirmation)
        _GAP_NEWS_BYPASS_PCT = 0.30  # 2026-06-30 14:33: raised from 0.25 — 3 consecutive hard_stop losses (AVEX -$348, AMBA -$358, PAVS -$549); tightening gap threshold
        _max_gh_today = self._long_cfg.max_gap_hold_entries_per_day
        if _max_gh_today > 0 and self._gap_hold_entry_count >= _max_gh_today:
            pass  # daily cap reached — fall through to short logic
        elif sym in self._gap_confirmed:
            # 2026-06-25: block re-entry after a same-day gap-hold loss.
            # CUPR entered 4× on 2026-06-24 and lost every time — once a ticker loses
            # via gap-hold, further entries have shown zero edge on the same day.
            if self._gap_hold_losses.get(sym, 0) >= 1:
                if sym not in self._gap_rejection_logged:
                    logger.info(
                        "GAP-HOLD blocked %s: already lost today via gap-hold (%d loss(es))",
                        sym, self._gap_hold_losses[sym],
                    )
                    self._gap_rejection_logged.add(sym)
            else:
                reg = self._regime
                if reg["date"] == today and reg["uptrend"]:
                    if (bar.close >= self._long_cfg.stage1_min_price
                            and baseline * 390 * bar.close >= self._long_cfg.min_avg_dollar_volume):
                        can_enter, ce_reason = self._long_portfolio.can_enter(sector="Unknown", now=bar.timestamp)
                        if can_enter:
                            prev_close = self._prev_closes.get(sym, 0.0)
                            day_open = self._day_opens.get(sym, 0.0)
                            gap_pct = (day_open - prev_close) / prev_close if prev_close > 0 else 0.0
                            has_cat = self._news_filter.has_catalyst(sym, today)
                            # Block if price has drifted >30% above day_open (gap momentum exhausted)
                            if day_open > 0 and bar.close > day_open * 1.30:
                                if sym not in self._gap_rejection_logged:
                                    logger.info(
                                        "GAP-HOLD blocked %s: price %.2f is %.1f%% above day_open %.2f"
                                        " — momentum exhausted",
                                        sym, bar.close, (bar.close / day_open - 1) * 100, day_open,
                                    )
                                    self._gap_rejection_logged.add(sym)
                            # Block if ATR is too compressed (<0.5% of price) — stop sizing degenerates
                            elif atr_val < bar.close * 0.005:
                                if sym not in self._gap_rejection_logged:
                                    logger.info(
                                        "GAP-HOLD blocked %s: ATR %.4f < 0.5%% of price %.2f"
                                        " — compressed, position sizing unreliable",
                                        sym, atr_val, bar.close,
                                    )
                                    self._gap_rejection_logged.add(sym)
                            # Change #1: momentum at entry. The entry bar must close above
                            # the prior bar — buy only when price is actually rising, not
                            # merely "holding". A flat/down entry bar re-evaluates next bar,
                            # so the stock can still enter once it ticks up.
                            elif (_GAP_ENTRY_REQUIRE_RISING and prev_bar is not None
                                    and bar.close <= prev_bar.close):
                                logger.info(
                                    "GAP-HOLD blocked %s: entry bar not rising (close %.4f <= prev %.4f)"
                                    " — momentum absent at entry",
                                    sym, bar.close, prev_bar.close,
                                )
                            elif gap_pct >= _GAP_NEWS_BYPASS_PCT:
                                if not has_cat:
                                    logger.info(
                                        "GAP-HOLD entering %s without catalyst (gap=%.1f%% ≥ %.0f%% bypass)",
                                        sym, gap_pct * 100, _GAP_NEWS_BYPASS_PCT * 100,
                                    )
                                self._enter_gap_hold(sym, bar, atr_val, baseline)
                                return
                            elif has_cat:
                                self._enter_gap_hold(sym, bar, atr_val, baseline)
                                return
                            else:
                                if sym not in self._gap_rejection_logged:
                                    logger.info(
                                        "GAP-HOLD blocked %s: no catalyst (gap=%.1f%% < %.0f%% bypass threshold)",
                                        sym, gap_pct * 100, _GAP_NEWS_BYPASS_PCT * 100,
                                    )
                                    self._gap_rejection_logged.add(sym)
                        else:
                            logger.debug("GAP-HOLD blocked %s: %s", sym, ce_reason)

        # --- Short entry (HOD-rejection, ETB-only, independent heat budget) ---
        # Fires when a parabolic mover (≥50% from day open) forms a multi-bar top.
        # The regime gate (SPY < 50-day MA) is checked here; no long position in same sym.
        if self._shorts_enabled and sym not in self._long_portfolio.positions:
            hod_state = self._short_hod.get(sym)
            if hod_state and hod_state["qualified"]:
                # Log once when sym is HOD-qualified but not ETB-shortable
                if sym not in self._etb_set:
                    logger.debug("HOD-REJECTION blocked %s: not in ETB set", sym)
                elif (bar.close >= self._short_cfg.stage1_min_price
                        and baseline * 390 * bar.close >= self._short_cfg.min_avg_dollar_volume):
                    reg = self._regime
                    if reg.get("short_allowed") and reg["date"] == today:
                        bars_since = hod_state["bars_since_hod"]
                        if bars_since >= self._short_hod_rej_bars:
                            can_enter, reason = self._short_portfolio.can_enter(sector="Unknown", now=bar.timestamp)
                            if can_enter:
                                self._enter_short_hod(sym, bar, atr_val, hod_state["hod"])
                                return
                            else:
                                logger.info("HOD-REJECTION blocked %s: %s (bars_since=%d)", sym, reason, bars_since)

        # --- Long entry (independent heat budget) ---
        # Regime is evaluated off the event-loop thread (see _maybe_refresh_regime);
        # here we only read the cached result. If it isn't current for today yet,
        # skip longs rather than block the bar handler on an SPY download.
        reg = self._regime
        if reg["date"] != today or not reg["uptrend"]:
            return
        if bar.close < self._long_cfg.stage1_min_price:
            return
        if baseline * 390 * bar.close < self._long_cfg.min_avg_dollar_volume:
            return

        # Gate order mirrors the backtest simulator: can_enter → validate() →
        # post-validate filters (day-high distance, volume acceleration). The
        # validator must be reached on the same bars as the backtest so its ROC
        # history isn't computed over a different (sparser) set of bars.
        can_enter, reason = self._long_portfolio.can_enter(sector="Unknown", now=bar.timestamp)
        if not can_enter:
            logger.debug("Long blocked for %s: %s", sym, reason)
            return
        if not self._long_validator.validate(bar, baseline):
            return

        # Post-validate filter 1: entry must be ≥ stage2_min_dist_from_day_high_pct
        # below the intraday high (avoid chasing the absolute top of a spike).
        day_high = self._day_highs.get(sym, bar.high)
        min_dist = self._long_cfg.stage2_min_dist_from_day_high_pct
        if min_dist > 0 and day_high > 0 and (day_high - bar.close) / day_high < min_dist:
            return

        # Post-validate filter 2: entry bar volume must be ≥ stage2_min_vol_vs_prev_bar
        # of the previous bar's volume (skip entries after the volume spike has peaked).
        vol_accel = getattr(self._long_cfg, "stage2_min_vol_vs_prev_bar", 0.0)
        if vol_accel > 0 and prev_bar is not None and prev_bar.volume > 0:
            if bar.volume < prev_bar.volume * vol_accel:
                return

        conf = self._long_validator.confidence_score(bar, baseline)
        mult = self._long_cfg.confidence_multiplier(conf)

        has_cat = self._news_filter.has_catalyst(sym, today)
        if not has_cat:
            # Tier-4 bypass: extreme confidence signal doesn't need external news.
            if mult >= _HYBRID_CHASE_TIER:
                logger.info("Tier-4 bypass for %s — entering without news catalyst", sym)
                has_cat = True
            else:
                logger.debug("No catalyst for %s — skipping entry", sym)
                return

        if mult >= _HYBRID_CHASE_TIER:
            # Tier-4: market entry with 4×ATR target (high confidence → no waiting)
            self._enter_long(sym, bar, atr_val, baseline)
        elif bar_time < _PULLBACK_LIMIT_CUTOFF:
            # Tier 1–3: arm a 2×ATR pullback limit (expires in 60 min or at 15:45)
            self._arm_pullback_limit(sym, bar, atr_val, mult)

    def _enter_short(self, sym: str, bar: Bar, atr_val: float, baseline: float) -> None:
        size = compute_position_size(self._equity, atr_val, bar.close, self._short_cfg)
        # Whole shares only: the broker stop order truncates to int, so a fractional
        # position would be partly unprotected.
        size.shares = int(size.shares)
        if size.shares <= 0:
            return
        self._entered_today.add(sym)
        fill_est = bar.close
        stop_price = size.short_stop(fill_est)
        target_price = size.short_target(fill_est)
        position = Position(
            ticker=sym, direction="short", shares=size.shares,
            entry_price=fill_est, stop_price=stop_price, target_price=target_price,
            entry_time=bar.timestamp, atr_at_entry=atr_val,
            signals=["momentum_short"], sector="Unknown",
            highest_close=fill_est, entry_bar_volume=bar.volume,
        )
        self._short_portfolio.add_position(position)
        with self._open_syms_lock:
            self._open_syms.add(sym)
        try:
            order_id = broker.short_sell(sym, size.shares)
        except Exception as exc:
            self._short_portfolio.remove_position(sym)
            self._entered_today.discard(sym)
            with self._open_syms_lock:
                self._open_syms.discard(sym)
            logger.error("Short sell failed for %s: %s", sym, exc)
            return
        signal_price = fill_est
        actual_fill = broker.get_fill_price(order_id)
        if actual_fill is not None and actual_fill != fill_est:
            logger.info(
                "Short fill for %s: est=%.2f actual=%.2f (slippage %.3f%%)",
                sym, fill_est, actual_fill,
                abs(actual_fill - fill_est) / fill_est * 100,
            )
            fill_est = actual_fill
            position.entry_price = actual_fill
            position.stop_price = size.short_stop(actual_fill)
            position.target_price = size.short_target(actual_fill)
            position.highest_close = actual_fill
        logger.info(
            "SHORT %s x%d @ %.2f stop=%.2f target=%.2f short_heat=%.1f%% order=%s",
            sym, size.shares, fill_est, position.stop_price, position.target_price,
            self._short_portfolio.portfolio_heat_pct * 100, order_id,
        )
        self._submit_broker_stop(sym, position)
        self._state.save_position(position)
        self._open_records[sym] = TradeRecord(
            ticker=sym, direction="short", entry_time=bar.timestamp,
            entry_price=fill_est, shares=size.shares,
            stop_price=position.stop_price, target_price=position.target_price,
            signals=["momentum_short"], sector="Unknown", regime="",
            portfolio_heat_at_entry=self._short_portfolio.portfolio_heat_pct,
            expected_slippage_pct=0.0,
            entry_slippage_pct=(fill_est - signal_price) / signal_price if signal_price else None,
        )
        self._sync_dash()

    def _enter_short_hod(self, sym: str, bar: Bar, atr_val: float, hod_price: float) -> None:
        """Short a parabolic reversal: stop above the day's high, target below entry.

        Unlike the momentum short (stop above entry), the stop here is anchored to the
        actual day high so a final squeeze above the entry doesn't immediately stop us out.
        """
        fill_est = bar.close
        stop_price = round(hod_price + _SHORT_HOD_STOP_MULT * atr_val, 2)
        target_price = round(fill_est - _SHORT_HOD_TARGET_MULT * atr_val, 2)
        if target_price <= 0 or stop_price <= fill_est:
            return
        stop_dist = stop_price - fill_est
        if stop_dist <= 0:
            return
        cfg = self._short_cfg
        shares = int(min(
            (self._equity * cfg.risk_per_trade) / stop_dist,
            (self._equity * cfg.max_position_pct) / fill_est,
        ))
        if shares <= 0:
            return

        self._entries_today += 1
        self._entered_today.add(sym)
        position = Position(
            ticker=sym, direction="short", shares=shares,
            entry_price=fill_est, stop_price=stop_price, target_price=target_price,
            entry_time=bar.timestamp, atr_at_entry=atr_val,
            signals=["hod_rejection"], sector="Unknown",
            highest_close=fill_est, entry_bar_volume=bar.volume,
        )
        self._short_portfolio.add_position(position)
        with self._open_syms_lock:
            self._open_syms.add(sym)
        try:
            order_id = broker.short_sell(sym, shares)
        except Exception as exc:
            self._short_portfolio.remove_position(sym)
            self._entered_today.discard(sym)
            with self._open_syms_lock:
                self._open_syms.discard(sym)
            logger.error("Short HOD sell failed for %s: %s", sym, exc)
            return

        signal_price = fill_est
        actual_fill = broker.get_fill_price(order_id)
        if actual_fill is not None and actual_fill != fill_est:
            logger.info(
                "Short HOD fill for %s: est=%.2f actual=%.2f (slippage %.3f%%)",
                sym, fill_est, actual_fill,
                abs(actual_fill - fill_est) / fill_est * 100,
            )
            fill_est = actual_fill
            position.entry_price = actual_fill
            position.stop_price = round(hod_price + _SHORT_HOD_STOP_MULT * atr_val, 2)
            position.target_price = round(actual_fill - _SHORT_HOD_TARGET_MULT * atr_val, 2)
            position.highest_close = actual_fill

        logger.info(
            "SHORT-HOD %s x%d @ %.2f  hod=%.2f  stop=%.2f  target=%.2f  "
            "short_heat=%.1f%%  order=%s",
            sym, shares, fill_est, hod_price,
            position.stop_price, position.target_price,
            self._short_portfolio.portfolio_heat_pct * 100, order_id,
        )
        self._submit_broker_stop(sym, position)
        self._state.save_position(position)
        self._open_records[sym] = TradeRecord(
            ticker=sym, direction="short", entry_time=bar.timestamp,
            entry_price=fill_est, shares=shares,
            stop_price=position.stop_price, target_price=position.target_price,
            signals=["hod_rejection"], sector="Unknown", regime="",
            portfolio_heat_at_entry=self._short_portfolio.portfolio_heat_pct,
            expected_slippage_pct=0.0,
            entry_slippage_pct=(fill_est - signal_price) / signal_price if signal_price else None,
        )
        self._sync_dash()

    def _enter_long(self, sym: str, bar: Bar, atr_val: float, baseline: float) -> None:
        cfg = self._long_cfg
        size = compute_position_size(self._equity, atr_val, bar.close, cfg, bar_volume=bar.volume)
        # Whole shares only: the broker stop order truncates to int, so a fractional
        # position would be partly unprotected.
        size.shares = int(size.shares)
        if size.shares <= 0:
            return
        # Confidence tiers: scale shares up, capped at max_position_pct (matches simulator).
        mult = cfg.confidence_multiplier(self._long_validator.confidence_score(bar, baseline))
        if mult > 1.0:
            max_shares = int(self._equity * cfg.max_position_pct / bar.close)
            size.shares = min(int(size.shares * mult), max(0, max_shares))
            logger.debug("Confidence %.1f× → %d shares for %s", mult, size.shares, sym)
        if size.shares <= 0:
            return
        self._entries_today += 1
        self._entered_today.add(sym)
        fill_est = bar.close
        stop_price = size.long_stop(fill_est)
        target_price = size.long_target(fill_est)
        position = Position(
            ticker=sym, direction="long", shares=size.shares,
            entry_price=fill_est, stop_price=stop_price, target_price=target_price,
            entry_time=bar.timestamp, atr_at_entry=atr_val,
            signals=["momentum_long"], sector="Unknown",
            highest_close=fill_est, entry_bar_volume=bar.volume,
        )
        self._long_portfolio.add_position(position)
        with self._open_syms_lock:
            self._open_syms.add(sym)
        try:
            order_id = broker.buy(sym, size.shares)
        except Exception as exc:
            self._long_portfolio.remove_position(sym)
            self._entered_today.discard(sym)
            with self._open_syms_lock:
                self._open_syms.discard(sym)
            logger.error("Long buy failed for %s: %s", sym, exc)
            return
        signal_price = fill_est
        actual_fill = broker.get_fill_price(order_id)
        if actual_fill is not None and actual_fill != fill_est:
            logger.info(
                "Long fill for %s: est=%.2f actual=%.2f (slippage %.3f%%)",
                sym, fill_est, actual_fill,
                abs(actual_fill - fill_est) / fill_est * 100,
            )
            fill_est = actual_fill
            position.entry_price = actual_fill
            position.stop_price = size.long_stop(actual_fill)
            position.target_price = size.long_target(actual_fill)
            position.highest_close = actual_fill
        logger.info(
            "LONG %s x%d @ %.2f stop=%.2f target=%.2f long_heat=%.1f%% order=%s",
            sym, size.shares, fill_est, position.stop_price, position.target_price,
            self._long_portfolio.portfolio_heat_pct * 100, order_id,
        )
        self._submit_broker_stop(sym, position)
        self._state.save_position(position)
        self._open_records[sym] = TradeRecord(
            ticker=sym, direction="long", entry_time=bar.timestamp,
            entry_price=fill_est, shares=size.shares,
            stop_price=position.stop_price, target_price=position.target_price,
            signals=["momentum_long"], sector="Unknown", regime="",
            portfolio_heat_at_entry=self._long_portfolio.portfolio_heat_pct,
            expected_slippage_pct=0.0,
            entry_slippage_pct=(fill_est - signal_price) / signal_price if signal_price else None,
        )
        self._sync_dash()

    def _enter_gap_hold(self, sym: str, bar: Bar, atr_val: float, baseline: float) -> None:
        cfg = self._long_cfg
        size = compute_position_size(self._equity, atr_val, bar.close, cfg, bar_volume=bar.volume)
        size.shares = int(size.shares)
        if size.shares <= 0:
            return
        self._gap_hold_entry_count += 1
        self._entries_today += 1
        self._entered_today.add(sym)
        fill_est = bar.close
        stop_price = size.long_stop(fill_est)
        target_price = size.long_target(fill_est)
        position = Position(
            ticker=sym, direction="long", shares=size.shares,
            entry_price=fill_est, stop_price=stop_price, target_price=target_price,
            entry_time=bar.timestamp, atr_at_entry=atr_val,
            signals=["gap_hold"], sector="Unknown",
            highest_close=fill_est, entry_bar_volume=bar.volume,
        )
        self._long_portfolio.add_position(position)
        with self._open_syms_lock:
            self._open_syms.add(sym)
        try:
            order_id = broker.buy(sym, size.shares)
        except Exception as exc:
            self._long_portfolio.remove_position(sym)
            self._entered_today.discard(sym)
            with self._open_syms_lock:
                self._open_syms.discard(sym)
            logger.error("Gap-hold buy failed for %s: %s", sym, exc)
            return
        signal_price = fill_est
        actual_fill = broker.get_fill_price(order_id)
        if actual_fill is not None and actual_fill != fill_est:
            logger.info(
                "Gap-hold fill for %s: est=%.2f actual=%.2f (slippage %.3f%%)",
                sym, fill_est, actual_fill,
                abs(actual_fill - fill_est) / fill_est * 100,
            )
            fill_est = actual_fill
            position.entry_price = actual_fill
            position.stop_price = size.long_stop(actual_fill)
            position.target_price = size.long_target(actual_fill)
            position.highest_close = actual_fill
        logger.info(
            "GAP-HOLD %s x%d @ %.2f stop=%.2f target=%.2f long_heat=%.1f%% order=%s",
            sym, size.shares, fill_est, position.stop_price, position.target_price,
            self._long_portfolio.portfolio_heat_pct * 100, order_id,
        )
        self._submit_broker_stop(sym, position)
        self._state.save_position(position)
        self._state.save_gap_losses(self._gap_hold_losses, self._entered_today)
        self._open_records[sym] = TradeRecord(
            ticker=sym, direction="long", entry_time=bar.timestamp,
            entry_price=fill_est, shares=size.shares,
            stop_price=position.stop_price, target_price=position.target_price,
            signals=["gap_hold"], sector="Unknown", regime="",
            portfolio_heat_at_entry=self._long_portfolio.portfolio_heat_pct,
            expected_slippage_pct=0.0,
            entry_slippage_pct=(fill_est - signal_price) / signal_price if signal_price else None,
        )
        self._sync_dash()

    def _arm_pullback_limit(self, sym: str, bar: Bar, atr_val: float, mult: float) -> None:
        """Submit a pullback limit buy order for a tier 1–3 signal and track it.

        The limit is placed _HYBRID_PULLBACK_ATR × ATR below the signal close.
        If unfilled after _HYBRID_PULLBACK_TTL_MIN minutes, it is cancelled.
        On fill, _open_pullback_position() creates the position with a 2.5×ATR target.
        """
        cfg = self._long_cfg
        limit_price = round(bar.close - _HYBRID_PULLBACK_ATR * atr_val, 2)
        if limit_price <= 0:
            return

        size = compute_position_size(self._equity, atr_val, limit_price, cfg)
        if mult > 1.0:
            max_shares = int(self._equity * cfg.max_position_pct / limit_price)
            size.shares = min(int(size.shares * mult), max(0, max_shares))
        if size.shares <= 0:
            return

        self._entered_today.add(sym)
        try:
            order_id = broker.submit_limit_order(sym, int(size.shares), "buy", limit_price)
        except Exception as exc:
            logger.error("Pullback limit order failed for %s: %s", sym, exc)
            self._entered_today.discard(sym)
            return

        self._armed_limits[sym] = {
            "order_id": order_id,
            "limit": limit_price,
            "atr": atr_val,
            "shares": int(size.shares),
            "mult": mult,
            "armed_at": bar.timestamp,
        }
        logger.info(
            "PULLBACK ARMED %s x%d limit=%.2f (%.1f×ATR below %.2f) ttl=%dmin order=%s",
            sym, int(size.shares), limit_price, _HYBRID_PULLBACK_ATR,
            bar.close, _HYBRID_PULLBACK_TTL_MIN, order_id,
        )

    def _open_pullback_position(self, sym: str, bar: Bar, armed: dict, fill_price: float, shares: int) -> None:
        """Register a position after a pullback limit order fills.

        `shares` is the broker-confirmed filled quantity (which may be less than
        the armed quantity for a partial fill), not the originally-armed size.
        """
        if shares <= 0:
            return
        cfg = self._long_cfg
        atr_val = armed["atr"]

        size = compute_position_size(self._equity, atr_val, fill_price, cfg)
        stop_price = size.long_stop(fill_price)
        target_price = round(fill_price + _HYBRID_PULLBACK_TARGET_ATR * atr_val, 2)

        position = Position(
            ticker=sym, direction="long", shares=shares,
            entry_price=fill_price, stop_price=stop_price, target_price=target_price,
            entry_time=bar.timestamp, atr_at_entry=atr_val,
            signals=["momentum_pullback"], sector="Unknown",
            highest_close=fill_price, entry_bar_volume=bar.volume,
        )
        self._long_portfolio.add_position(position)
        with self._open_syms_lock:
            self._open_syms.add(sym)

        self._submit_broker_stop(sym, position)
        self._state.save_position(position)
        self._open_records[sym] = TradeRecord(
            ticker=sym, direction="long", entry_time=bar.timestamp,
            entry_price=fill_price, shares=shares,
            stop_price=stop_price, target_price=target_price,
            signals=["momentum_pullback"], sector="Unknown", regime="",
            portfolio_heat_at_entry=self._long_portfolio.portfolio_heat_pct,
            expected_slippage_pct=0.0,
        )
        logger.info(
            "PULLBACK FILL %s x%d @ %.2f stop=%.2f target=%.2f (%.1f×ATR) heat=%.1f%%",
            sym, shares, fill_price, stop_price, target_price,
            _HYBRID_PULLBACK_TARGET_ATR, self._long_portfolio.portfolio_heat_pct * 100,
        )
        self._sync_dash()

    def _process_armed_limit(self, sym: str, bar: Bar, bar_time: dtime) -> None:
        """Resolve a resting pullback limit order against the broker's order state.

        Fills are confirmed via the order's status/filled_qty rather than inferred
        from bar prices — a bar touching the limit doesn't guarantee a fill, and an
        order can fill between bars. This prevents both phantom positions (registering
        a fill that didn't happen) and orphaned positions (a real fill we never track).
        """
        armed = self._armed_limits[sym]
        try:
            status, filled_qty, avg_price = broker.get_order_status(armed["order_id"])
        except Exception as exc:
            logger.warning(
                "Could not query pullback order %s for %s: %s — will retry next bar",
                armed["order_id"], sym, exc,
            )
            return

        # Fully filled → register the position with the actual fill price/qty.
        if "filled" in status and "partially" not in status:
            del self._armed_limits[sym]
            fill_price = avg_price if avg_price is not None else armed["limit"]
            self._open_pullback_position(sym, bar, armed, fill_price, filled_qty or armed["shares"])
            return

        # Otherwise decide whether to cancel: kill switch, TTL, or EOD cutoff.
        elapsed_min = (bar.timestamp - armed["armed_at"]).total_seconds() / 60
        ks_active = self._long_portfolio.kill_switch_active
        if not (ks_active or bar_time >= _PULLBACK_LIMIT_CUTOFF or elapsed_min >= _HYBRID_PULLBACK_TTL_MIN):
            return  # still working and within its window — leave it resting

        try:
            broker.cancel_order(armed["order_id"])
        except Exception as exc:
            logger.debug("Pullback limit cancel %s: %s", sym, exc)
        # Re-query after cancel: a fill may have raced the cancel. Never orphan real shares.
        try:
            _, filled_qty, avg_price = broker.get_order_status(armed["order_id"])
        except Exception:
            filled_qty, avg_price = 0, None
        del self._armed_limits[sym]
        if filled_qty > 0:
            fill_price = avg_price if avg_price is not None else armed["limit"]
            logger.info("Pullback %s had %d filled shares at cancel — registering position", sym, filled_qty)
            self._open_pullback_position(sym, bar, armed, fill_price, filled_qty)
        else:
            why = "kill_switch" if ks_active else ("eod_cutoff" if bar_time >= _PULLBACK_LIMIT_CUTOFF else "ttl")
            logger.info("Pullback limit cancelled unfilled for %s (%s)", sym, why)

    def _cancel_all_armed_limit_orders(self, reason: str) -> None:
        """Best-effort cancel of every resting pullback order (e.g. on kill switch).

        Entries are left in _armed_limits so the next bar's _process_armed_limit
        reconciles any race-fill before dropping them.
        """
        for sym, armed in list(self._armed_limits.items()):
            try:
                broker.cancel_order(armed["order_id"])
                logger.info("Cancelled resting pullback order for %s (%s)", sym, reason)
            except Exception:
                pass

    def _execute_close(self, sym: str, fill_price: float, reason: str, portfolio: PortfolioState) -> None:
        """Close a position and account for it using the ACTUAL broker fill price.

        `fill_price` is the theoretical exit (stop/target/last price); the real
        fill is fetched from the exit order (or the broker stop, if it fired) and
        used for PnL, the trade record, and slippage tracking. Equity is synced
        only after the close is confirmed, so a failed exit leaves it untouched.
        """
        with self._close_lock:
            position = portfolio.remove_position(sym)
        if not position:
            return
        with self._open_syms_lock:
            self._open_syms.discard(sym)

        is_short = position.direction == "short"

        # Cancel broker stop before covering.
        # Returns False if the broker already filled the stop (covered/sold for us).
        need_to_cover = self._cancel_broker_stop(sym, position)

        actual_fill = fill_price  # fallback if the broker doesn't report a fill price
        if need_to_cover:
            try:
                order_id = broker.buy_to_cover(sym, position.shares) if is_short else broker.sell(sym, position.shares)
            except Exception as exc:
                # Restore position — neither the exit nor the stop order worked.
                # No equity change has been applied yet, so nothing to undo.
                portfolio.add_position(position)
                with self._open_syms_lock:
                    self._open_syms.add(sym)
                logger.error("Exit order failed for %s: %s — position restored", sym, exc)
                return
            got = broker.get_fill_price(order_id)
            if got is not None:
                actual_fill = got
            self._state.remove_position(sym)
            action = "COVER" if is_short else "SELL"
            logger.info(
                "%s %s x%d @ %.2f (theo %.2f) reason=%s order=%s",
                action, sym, position.shares, actual_fill, fill_price, reason, order_id,
            )
        else:
            # Broker stop already filled — use its actual fill price if available.
            if position.stop_order_id:
                got = broker.get_fill_price(position.stop_order_id, max_wait_sec=0.5)
                if got is not None:
                    actual_fill = got
            self._state.remove_position(sym)
            action = "COVERED" if is_short else "SOLD"
            logger.info(
                "%s by broker stop %s x%d @ %.2f reason=%s",
                action, sym, position.shares, actual_fill, reason,
            )

        # Accounting under _close_lock so the equity/slippage/record updates are
        # atomic across the event-loop thread and the EOD-timer thread (both call
        # _execute_close). Broker I/O above is intentionally outside the lock.
        closed_record: Optional[TradeRecord] = None
        with self._close_lock:
            # PnL from the actual fill price.
            if is_short:
                pnl = round((position.entry_price - actual_fill) * position.shares, 2)
            else:
                pnl = round((actual_fill - position.entry_price) * position.shares, 2)
            self._sync_equity(self._equity + pnl)

            # Update consecutive loss counter on the relevant portfolio.
            if pnl < 0:
                portfolio.consecutive_losses += 1
            else:
                portfolio.consecutive_losses = 0

            logger.info(
                "CLOSED %s pnl=%.2f reason=%s equity=%.2f", sym, pnl, reason, self._equity,
            )

            record = self._open_records.pop(sym, None)
            # Track same-day gap-hold losses per symbol to block re-entry (see _gap_hold_losses)
            if pnl < 0 and record is not None and "gap_hold" in (record.signals or []):
                self._gap_hold_losses[sym] = self._gap_hold_losses.get(sym, 0) + 1
                self._state.save_gap_losses(self._gap_hold_losses, self._entered_today)
            if record is not None:
                record.exit_time = datetime.now(_ET)
                record.exit_price = actual_fill
                record.pnl = pnl
                record.exit_reason = reason
                if position.entry_price > 0:
                    record.actual_slippage_pct = (actual_fill - fill_price) / position.entry_price
                self._trade_logger.log(record)
                closed_record = record
        self._sync_dash(closed_record=closed_record)

    # ------------------------------------------------------------------
    # Startup reconciliation
    # ------------------------------------------------------------------

    def _reconcile_existing_positions(self) -> List[str]:
        """Cross-reference Alpaca's open positions with today's saved state.

        - Positions in Alpaca + saved state  → restore with full stop/target/ATR
        - Positions in Alpaca, no saved state → reconstruct with 2% ATR approximation
        - Positions in saved state, not in Alpaca → broker stop fired; clean up state
        - Fresh broker stop re-submitted for every reconciled position

        Returns the list of symbols that need bar stream subscriptions.
        """
        try:
            alpaca_positions = broker.get_all_positions_detail()
        except Exception as exc:
            logger.error("Could not fetch Alpaca positions for reconciliation: %s", exc)
            return []

        saved = self._state.get_saved_positions()

        # Clean up saved entries that Alpaca has already closed (broker stop fired)
        for sym in list(saved.keys()):
            if sym not in alpaca_positions:
                logger.info(
                    "Saved position %s not found in Alpaca — "
                    "likely covered by broker stop while offline. Removing from state.",
                    sym,
                )
                self._state.remove_position(sym)

        if not alpaca_positions:
            return []

        logger.info(
            "Reconciling %d Alpaca position(s) against %d saved entry(s)",
            len(alpaca_positions), len(saved),
        )

        syms_to_subscribe: List[str] = []
        for sym, details in alpaca_positions.items():
            side = details["side"]
            qty = details["qty"]
            entry_price = details["entry_price"]

            position = self._state.restore_position(sym)
            if position is not None:
                if position.shares != qty:
                    logger.warning(
                        "Share count mismatch for %s: saved=%d alpaca=%d — using Alpaca value",
                        sym, position.shares, qty,
                    )
                    position.shares = qty
                logger.info(
                    "Restored %s x%d @ %.2f (stop=%.2f target=%.2f) from saved state",
                    sym, qty, entry_price, position.stop_price, position.target_price,
                )
            else:
                direction = "short" if side == "short" else "long"
                approx_atr = entry_price * 0.02
                stop_price = round(entry_price + approx_atr, 2) if direction == "short" else round(entry_price - approx_atr, 2)
                target_price = round(entry_price - 2 * approx_atr, 2) if direction == "short" else round(entry_price + 2 * approx_atr, 2)
                position = Position(
                    ticker=sym, direction=direction, shares=qty,
                    entry_price=entry_price, stop_price=stop_price, target_price=target_price,
                    entry_time=datetime.now(_ET), atr_at_entry=approx_atr,
                    signals=["reconciled"], sector="Unknown",
                    highest_close=entry_price, entry_bar_volume=0,
                )
                logger.warning(
                    "No saved state for %s x%d @ %.2f — reconstructed %s approx "
                    "stop=%.2f target=%.2f (2%% ATR)",
                    sym, qty, entry_price, direction, stop_price, target_price,
                )

            # Route into the correct portfolio based on direction
            target_portfolio = self._short_portfolio if position.direction == "short" else self._long_portfolio

            # Re-submit a fresh broker stop (DAY orders expire; prior one may be stale)
            position.stop_order_id = ""
            self._submit_broker_stop(sym, position)
            self._state.save_position(position)

            target_portfolio.add_position(position)
            self._entered_today.add(sym)
            with self._open_syms_lock:
                self._open_syms.add(sym)
            syms_to_subscribe.append(sym)

        return syms_to_subscribe

    # ------------------------------------------------------------------
    # Watchlist refresh — daemon thread
    # ------------------------------------------------------------------

    def _build_watchlist_from_movers(self, movers: List[dict]) -> Set[str]:
        # Use the lower of the two price floors so both strategies see relevant symbols
        min_price = min(self._short_cfg.stage1_min_price, self._long_cfg.stage1_min_price)
        # Derive the mover threshold from config (long strategy is the active one) so it
        # can't drift away from stage1_min_price_change_pct.
        min_change_pct = self._long_cfg.stage1_min_price_change_pct * 100
        watchlist: Set[str] = set()
        for m in movers:
            sym = m.get("symbol", "")
            pct = m.get("percent_change", 0.0)
            price = m.get("price", 0.0)
            if pct < min_change_pct:
                continue
            if price < min_price:
                continue
            if "." in sym:
                # Skip rights/warrants/when-issued (e.g. AMPG.RT.A) — IBKR
                # cannot reliably resolve these as Stock contracts.
                logger.debug("Skipping non-standard ticker %s", sym)
                continue
            watchlist.add(sym)
        return watchlist

    def _maybe_refresh_regime(self, today: date) -> None:
        """Evaluate the SPY regime for `today` if not already current. Off the bar thread.

        Runs in the watchlist daemon (and once at startup) so the event-loop thread
        never blocks on the SPY download inside _on_bar. Clears the in-process SPY
        cache on a day change so a 24/7 process re-reads fresh data each day.
        The result dict is swapped in atomically; _on_bar only reads it.
        """
        if self._regime["date"] == today:
            return
        from bot.data import regime as _regime_mod
        _regime_mod.clear_spy_cache()
        uptrend = True        # regime filter bypassed — paper trading mode
        short_allowed = True  # regime filter bypassed — paper trading mode
        self._regime = {"date": today, "uptrend": uptrend, "short_allowed": short_allowed}
        if uptrend:
            logger.info("REGIME %s: SPY uptrend — long entries enabled", today)
        else:
            logger.info(
                "REGIME %s: SPY below %d-day MA — long entries blocked",
                today, self._regime_filter._ma_period,
            )
        if short_allowed:
            logger.info("REGIME %s: SPY below %d-day MA — short HOD entries enabled", today,
                        self._short_regime_filter._ma_period)
        else:
            logger.info("REGIME %s: SPY above %d-day MA — short HOD entries blocked",
                        today, self._short_regime_filter._ma_period)
        # Update only the regime fields here (not full _sync_dash): this runs on the
        # daemon thread, and rebuilding the positions snapshot would iterate the
        # portfolio dicts concurrently with the event-loop thread mutating them.
        if self._dash is not None:
            with self._dash._lock:
                self._dash.regime_uptrend = uptrend
                self._dash.regime_date = today
                self._dash.short_allowed = short_allowed

    def _warm_news_cache(self, symbols: Set[str], today: date) -> None:
        """Pre-fetch catalyst news for watchlist symbols so _on_bar reads from cache.

        Runs in the watchlist daemon. has_catalyst() memoizes per (symbol, date), so
        each symbol is fetched at most once per day; on a date change the warmed set
        is reset so the new day's catalysts are fetched.
        """
        if self._news_warmed_date != today:
            self._news_warmed = set()
            self._news_warmed_date = today
        for sym in symbols:
            if sym in self._news_warmed:
                continue
            try:
                self._news_filter.has_catalyst(sym, today)  # populates the in-memory + disk cache
            except Exception as exc:
                logger.debug("News pre-warm failed for %s: %s", sym, exc)
            self._news_warmed.add(sym)

    def _refresh_watchlist_loop(self) -> None:
        """Periodically fetch movers and queue sub/unsub requests. Daemon thread.

        Also refreshes the SPY regime and pre-warms catalyst news here so the
        event-loop thread (_on_bar) never blocks on those network calls.
        """
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=_WATCHLIST_REFRESH_SEC)
            if self._stop_event.is_set():
                break

            # Keep the regime current (handles day rollover for a 24/7 process).
            try:
                self._maybe_refresh_regime(datetime.now(_ET).date())
            except Exception as exc:
                logger.warning("Regime refresh failed: %s", exc)

            try:
                movers = broker_ibkr.get_movers(
                    self._ibkr_host, self._ibkr_port,
                    self._ibkr_scanner_client_id, top_n=50,
                )
            except Exception as exc:
                logger.warning("Watchlist refresh failed: %s", exc)
                continue

            new_watchlist = self._build_watchlist_from_movers(movers)

            with self._open_syms_lock:
                protected = set(self._open_syms)

            # Symbols to add: newly qualifying movers not yet subscribed today
            to_add = new_watchlist - self._current_watchlist

            # Never remove a symbol that qualified today — matches backtest screener behaviour
            # where a stock that hit the 15% threshold at any point stays in the candidate
            # pool for the full session, even if it pulls back below 15% later.
            # Only remove symbols that never qualified today AND have no open position.
            to_remove = (self._current_watchlist - new_watchlist - self._subscribed_today) - protected

            if self._stream is not None:
                with self._baseline_vols_lock:
                    new_without_baseline = [s for s in to_add if s not in self._baseline_vols]
                if new_without_baseline:
                    self._fetch_baseline_volumes(new_without_baseline)
                for sym in to_add:
                    self._stream.subscribe(sym)
                    self._subscribed_today.add(sym)
                for sym in to_remove:
                    self._stream.unsubscribe(sym)

            self._current_watchlist = (self._current_watchlist | to_add) - to_remove
            if to_add or to_remove:
                logger.info(
                    "Watchlist updated: +%d -%d (total=%d, held-today=%d)",
                    len(to_add), len(to_remove), len(self._current_watchlist),
                    len(self._subscribed_today),
                )

            # Pre-warm catalyst news for the current watchlist (one fetch per symbol
            # per day) so _on_bar's has_catalyst() check hits cache, never the network.
            if self._current_watchlist:
                self._warm_news_cache(self._current_watchlist, datetime.now(_ET).date())

    # ------------------------------------------------------------------
    # Status log — daemon thread
    # ------------------------------------------------------------------

    def _status_log_loop(self) -> None:
        """Log equity, open positions, and watchlist size every minute. Daemon thread."""
        _bar_flush_ticks = 0
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=_STATUS_LOG_SEC)
            if self._stop_event.is_set():
                break
            with self._open_syms_lock:
                open_syms = set(self._open_syms)
            logger.info(
                "STATUS equity=%.2f short_positions=%d short_heat=%.1f%% "
                "long_positions=%d long_heat=%.1f%% watchlist=%d",
                self._equity,
                len(self._short_portfolio.positions),
                self._short_portfolio.portfolio_heat_pct * 100,
                len(self._long_portfolio.positions),
                self._long_portfolio.portfolio_heat_pct * 100,
                len(self._current_watchlist),
            )
            for sym in open_syms:
                pos = self._short_portfolio.positions.get(sym) or self._long_portfolio.positions.get(sym)
                if pos:
                    logger.info(
                        "  [%s] %s %d @ %.2f | stop=%.2f | target=%.2f | broker_stop=%s",
                        sym, pos.direction, pos.shares, pos.entry_price,
                        pos.stop_price, pos.target_price,
                        pos.stop_order_id or "NONE",
                    )
            # Flush bar cache to disk every 5 minutes so files appear during the session
            _bar_flush_ticks += 1
            if _bar_flush_ticks >= 5:
                self._flush_bar_cache(overwrite_today=True)
                _bar_flush_ticks = 0

    # ------------------------------------------------------------------
    # Baseline volume helpers
    # ------------------------------------------------------------------

    def _fetch_baseline_volumes(self, symbols: List[str], max_attempts: int = 3) -> None:
        """Fetch baseline volume/prev_close/MA10 from yfinance, retrying failed
        symbols with exponential backoff.

        2026-07-01: a symbol failing here silently (yfinance is an unofficial,
        rate-limit-prone API) leaves it with no prev_close, so gap-hold can never
        detect a gap for it — this was previously a silent, unretried failure mode.
        Callers of this method already run off the event-loop thread (see class
        docstring), so blocking here for backoff is safe.
        """
        import yfinance as yf
        import pandas as pd

        today = date.today()
        start = (today - timedelta(days=35)).isoformat()
        end = today.isoformat()

        vol_updates: Dict[str, float] = {}
        ma10_updates: Dict[str, float] = {}
        prev_close_updates: Dict[str, float] = {}
        remaining = list(symbols)
        backoff = 1.0

        for attempt in range(1, max_attempts + 1):
            if not remaining:
                break
            try:
                raw = yf.download(
                    tickers=remaining, interval="1d", start=start, end=end,
                    progress=False, auto_adjust=False, group_by="ticker", threads=False,
                )
            except Exception as exc:
                logger.warning(
                    "Baseline volume fetch attempt %d/%d failed for %d symbols: %s",
                    attempt, max_attempts, len(remaining), exc,
                )
                if attempt < max_attempts:
                    time.sleep(backoff)
                    backoff *= 2
                continue

            still_missing: List[str] = []
            for sym in remaining:
                try:
                    df = raw if len(remaining) == 1 else (
                        raw[sym] if sym in raw.columns.get_level_values(0) else pd.DataFrame()
                    )
                    if df.empty:
                        still_missing.append(sym)
                        continue
                    avg_vol = float(df["Volume"].tail(20).mean())
                    vol_updates[sym] = avg_vol / 390
                    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
                    closes = df[close_col].dropna()
                    # 10-day close MA for the narrow-day filter
                    ma10 = float(closes.tail(_HYBRID_MA_PERIOD).mean())
                    if ma10 > 0:
                        ma10_updates[sym] = ma10
                    # Most recent past close for gap-hold gap detection
                    if not closes.empty:
                        pc = float(closes.iloc[-1])
                        if pc > 0:
                            prev_close_updates[sym] = pc
                    else:
                        still_missing.append(sym)
                except Exception:
                    still_missing.append(sym)

            remaining = still_missing
            if remaining and attempt < max_attempts:
                logger.info(
                    "Baseline volume fetch attempt %d/%d: %d/%d symbols still missing"
                    " (%s), retrying in %.0fs",
                    attempt, max_attempts, len(remaining), len(symbols), remaining, backoff,
                )
                time.sleep(backoff)
                backoff *= 2

        if vol_updates:
            with self._baseline_vols_lock:
                self._baseline_vols.update(vol_updates)
                self._daily_ma10.update(ma10_updates)
        if prev_close_updates:
            self._prev_closes.update(prev_close_updates)
        logger.info("Baseline volumes loaded for %d/%d symbols", len(vol_updates), len(symbols))
        if remaining:
            logger.warning(
                "Baseline volumes STILL MISSING after %d attempts for %d/%d symbols: %s",
                max_attempts, len(remaining), len(symbols), remaining,
            )

    def load_baseline_volumes(self, symbols: List[str]) -> None:
        logger.info("Fetching baseline volumes for %d initial symbols...", len(symbols))
        self._fetch_baseline_volumes(symbols)

    def _flush_bar_cache(self, overwrite_today: bool = False) -> None:
        """Write accumulated IBKR bars to the backtest cache directory.

        Each file is dated from the bars' own timestamps, not date.today(): the
        day-rollover flush writes the *previous* day's bars, so today() would
        mislabel them.

        overwrite_today=True  — mid-session flush: always overwrites today's file
                                so bars accumulate progressively on disk.
        overwrite_today=False — EOD/rollover flush: skips existing files so we
                                don't clobber Alpaca data for past dates.
        """
        today = date.today()
        written = skipped = 0
        for sym, bars in self._live_bar_cache.items():
            if not bars:
                continue
            try:
                bar_date = datetime.fromisoformat(bars[0]["t"]).astimezone(_ET).date()
            except Exception:
                bar_date = self._trading_date or today
            cache_path = self._bar_cache_dir / f"{sym}_{bar_date}.json"
            if cache_path.exists() and not (overwrite_today and bar_date == today):
                skipped += 1
                continue
            cache_path.write_text(json.dumps(bars))
            written += 1
        logger.info(
            "IBKR bar cache flushed: %d files written, %d skipped (already cached)",
            written, skipped,
        )
        if not overwrite_today:
            self._live_bar_cache.clear()

    # ------------------------------------------------------------------
    # ETB refresh — daemon thread (fix: ETB list loaded once at startup)
    # ------------------------------------------------------------------

    def _etb_refresh_loop(self) -> None:
        """Refresh the ETB list every 30 minutes. Daemon thread.

        The ETB set is an attribute reference, so replacing it atomically is
        GIL-safe — _on_bar's `sym in self._etb_set` always sees a complete set.
        """
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=_ETB_REFRESH_SEC)
            if self._stop_event.is_set():
                break
            try:
                new_etb = broker.get_etb_set()
                if new_etb:
                    self._etb_set = new_etb
                    logger.info("ETB set refreshed: %d shortable symbols", len(new_etb))
            except Exception as exc:
                logger.warning("ETB refresh failed (keeping previous list): %s", exc)

    # ------------------------------------------------------------------
    # EOD timer — daemon thread (fix: force-close even if no bar arrives)
    # ------------------------------------------------------------------

    def _eod_timer_loop(self) -> None:
        """At 15:55:30 ET, force-close any positions not already exited by bar events.

        This is a safety net for positions in halted or illiquid stocks that stop
        generating 1-min bars before 15:55, which would never trigger the bar-based
        EOD close in _on_bar. Runs in a daemon thread; does not block _on_bar.
        """
        while not self._stop_event.is_set():
            now_et = datetime.now(_ET)
            target = now_et.replace(hour=15, minute=55, second=30, microsecond=0)
            if now_et >= target:
                target = target + timedelta(days=1)
            wait_secs = (target - now_et).total_seconds()
            self._stop_event.wait(timeout=wait_secs)
            if self._stop_event.is_set():
                break
            self._submit_eod_close_orders()

    def _submit_eod_close_orders(self) -> None:
        """Force-close all open positions at market. Called from the EOD timer thread.

        Routes through _execute_close so these closes get the same accounting as any
        other exit — PnL, equity sync, trade-log record, slippage, dashboard update.
        _execute_close claims each position under _close_lock, so this can't double-
        close against a late bar event on the event-loop thread.
        """
        short_syms = list(self._short_portfolio.positions.keys())
        long_syms = list(self._long_portfolio.positions.keys())
        if not short_syms and not long_syms:
            return
        logger.info(
            "EOD timer: force-closing %d short + %d long position(s) at market",
            len(short_syms), len(long_syms),
        )
        for sym in short_syms:
            pos = self._short_portfolio.positions.get(sym)
            if pos is None:
                continue
            price = self._last_prices.get(sym, pos.entry_price)
            self._execute_close(sym, price, "eod_force", self._short_portfolio)
        for sym in long_syms:
            pos = self._long_portfolio.positions.get(sym)
            if pos is None:
                continue
            price = self._last_prices.get(sym, pos.entry_price)
            self._execute_close(sym, price, "eod_force", self._long_portfolio)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        # --- Step 1: Reconcile positions from previous run ---
        reconciled_syms = self._reconcile_existing_positions()
        if reconciled_syms:
            logger.info(
                "Reconciled %d position(s) from previous run: %s",
                len(reconciled_syms), reconciled_syms,
            )

        # --- Step 2: Fetch initial mover watchlist ---
        initial_watchlist: Set[str] = set()
        try:
            movers = broker_ibkr.get_movers(
                self._ibkr_host, self._ibkr_port,
                self._ibkr_scanner_client_id, top_n=50,
            )
            initial_watchlist = self._build_watchlist_from_movers(movers)
            logger.info("Movers endpoint: %d qualifying symbols", len(initial_watchlist))
        except Exception as exc:
            logger.error(
                "get_movers() failed: %s — starting with empty watchlist; "
                "refresh loop will retry every %ds.",
                exc, _WATCHLIST_REFRESH_SEC,
            )

        all_initial = initial_watchlist | set(reconciled_syms)

        # --- Step 3: Load baseline volumes ---
        if all_initial:
            self.load_baseline_volumes(list(all_initial))

        self._current_watchlist = initial_watchlist
        logger.info(
            "Starting BarStream on %d symbols (%d reconciled + %d movers)",
            len(all_initial), len(reconciled_syms), len(initial_watchlist),
        )

        # Evaluate the regime up front so the first bars see a current value rather
        # than blocking the event-loop thread on the SPY download (also keeps the
        # daemon from racing the open). Pre-warm catalysts for the initial watchlist.
        try:
            self._maybe_refresh_regime(datetime.now(_ET).date())
        except Exception as exc:
            logger.warning("Initial regime evaluation failed: %s — longs gated until refreshed", exc)
        if initial_watchlist:
            self._warm_news_cache(initial_watchlist, datetime.now(_ET).date())

        self._stream = BarStream(
            self._ibkr_host, self._ibkr_port, self._ibkr_client_id, list(all_initial),
            bar_size_seconds=self._long_cfg.bar_size_seconds,
        )
        self._stream.set_handler(self._on_bar)

        if self._dash is not None:
            from bot.dashboard.server import start_server
            self._sync_dash()
            start_server(self._dash)

        daemons = [
            ("watchlist-refresh", self._refresh_watchlist_loop),
            ("status-log", self._status_log_loop),
            ("eod-timer", self._eod_timer_loop),
        ]
        # Only refresh the ETB (shortable) list when shorts are actually enabled —
        # otherwise it would silently repopulate the intentionally-empty ETB set and
        # bring the short strategy back to life mid-session.
        if self._shorts_enabled:
            daemons.append(("etb-refresh", self._etb_refresh_loop))
        for name, target in daemons:
            threading.Thread(target=target, daemon=True, name=name).start()

        try:
            self._stream.run_with_reconnect(self._stop_event)
        finally:
            self._stop_event.set()
            self._flush_bar_cache()
            all_open = {**self._short_portfolio.positions, **self._long_portfolio.positions}
            logger.info(
                "Session ended. Equity: $%.2f | Open positions: %d",
                self._equity, len(all_open),
            )
            if all_open:
                logger.warning(
                    "Unclosed positions at shutdown: %s — "
                    "broker stops are active; they will be reconciled on next startup.",
                    list(all_open.keys()),
                )
