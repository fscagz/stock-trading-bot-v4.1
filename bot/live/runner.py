"""
V4 live runner — short and long momentum strategies with independent heat budgets.

On each 1-min bar:
  1. Update ATR / VWAP indicators
  2. If position open: check stop/target/manager exits → cancel broker stop → cover/sell
  3. If no position, window 9:45-11:30: run Stage-2 validator → enter → submit broker stop
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
After every short_sell, a stop-buy order is submitted to Alpaca at the same
stop price. If the process dies, Alpaca's stop covers the position automatically.
When our code exits a position for any reason, it cancels the broker stop first.
If the cancel fails because the stop already filled (broker covered while we were
monitoring), the cover order is skipped — preventing a double-buy.
When the trailing stop moves, the old broker stop is cancelled and a new one
submitted at the updated price.

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
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import copy

import bot.broker_alpaca as broker
import bot.broker_ibkr as broker_ibkr
from bot.backtest.news_filter import NewsFilter
from bot.config import V4Config, make_long_config
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

_ET = ZoneInfo("America/New_York")
_ENTRY_START = dtime(9, 30)  # entries open from market open
_EOD_FORCE_CLOSE = dtime(15, 55)
_WATCHLIST_REFRESH_SEC = 30
_STATUS_LOG_SEC = 60
_ETB_REFRESH_SEC = 1800        # refresh ETB list every 30 minutes
_MIN_MOVER_PCT = 15.0          # matches make_long_config() stage1_min_price_change_pct (0.15 = 15%)

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
        self._api_key = api_key
        self._secret_key = secret_key

        self._news_filter = NewsFilter(api_key, secret_key, cache_only=False)
        self._regime_filter = RegimeFilter()
        self._regime: dict = {"date": None, "uptrend": True}  # re-evaluated each trading day
        self._dash = dash
        self._trade_logger = TradeLogger(log_dir=trade_log_dir)
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
        # Reconcile equity with broker before resetting session_start_equity
        # so daily_pnl_pct() is accurate for the new day.
        try:
            actual_equity = broker.get_account_info()["portfolio_value"]
            self._sync_equity(actual_equity)
            logger.info("Daily equity sync: $%.2f", actual_equity)
        except Exception as exc:
            logger.warning("Could not sync equity at day rollover: %s — using last known value", exc)

        for portfolio in (self._long_portfolio, self._short_portfolio):
            portfolio.session_start_equity  = portfolio.equity
            portfolio.kill_switch_active    = False
            portfolio.consecutive_losses    = 0
            portfolio.cooldown_until        = None
            portfolio.session_slippage_actual   = 0.0
            portfolio.session_slippage_expected = 0.0

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
            self._dash.positions = {
                **self._short_portfolio.positions,
                **self._long_portfolio.positions,
            }
            # Use long portfolio for heat/kill-switch (shorts are disabled)
            lp = self._long_portfolio
            self._dash.portfolio_heat_pct = lp.portfolio_heat_pct
            self._dash.kill_switch_active = lp.kill_switch_active
            self._dash.consecutive_losses = lp.consecutive_losses
            self._dash.cooldown_until = lp.cooldown_until
            self._dash.regime_uptrend = self._regime["uptrend"]
            self._dash.regime_date = self._regime["date"]
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
            self._atrs[sym] = ATRIndicator(period=14)
            self._vwaps[sym] = VWAPIndicator()
        return self._atrs[sym], self._vwaps[sym]

    # ------------------------------------------------------------------
    # Broker stop management (event loop thread only)
    # ------------------------------------------------------------------

    def _submit_broker_stop(self, sym: str, position: Position) -> None:
        """Submit a broker-side stop-buy order and store the order ID."""
        try:
            order_id = broker.submit_stop_buy_order(sym, position.shares, position.stop_price)
            position.stop_order_id = order_id
            logger.debug("Broker stop submitted for %s @ %.2f (order %s)", sym, position.stop_price, order_id)
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

    def _replace_broker_stop(self, sym: str, position: Position) -> None:
        """Cancel old broker stop and submit a new one at the current stop price."""
        old_id = position.stop_order_id
        if old_id:
            try:
                broker.cancel_order(old_id)
            except Exception as exc:
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
            logger.info("New trading day %s — daily state reset", today)

        # Track running intraday high for each symbol (for stage2_min_dist_from_day_high_pct)
        if bar.high > self._day_highs.get(sym, 0.0):
            self._day_highs[sym] = bar.high

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
        self._last_prices[sym] = bar.close
        if self._dash is not None:
            with self._dash._lock:
                self._dash.last_prices[sym] = bar.close
        with self._baseline_vols_lock:
            baseline = self._baseline_vols.get(sym, 0.0)

        # Determine which portfolio owns this symbol (if any)
        active_portfolio = None
        if sym in self._short_portfolio.positions:
            active_portfolio = self._short_portfolio
        elif sym in self._long_portfolio.positions:
            active_portfolio = self._long_portfolio

        # Force-close all positions at 15:55 ET
        if bar_time >= _EOD_FORCE_CLOSE:
            if active_portfolio is not None:
                logger.info("EOD force-close %s", sym)
                self._execute_close(sym, bar.close, "eod_force", active_portfolio)
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
                        self._replace_broker_stop(sym, position)
                    else:
                        self._state.update_stop(sym, position.stop_price, position.stop_order_id)
            return

        # --- Entry logic ---
        if bar_time < _ENTRY_START:
            return
        if sym in self._entered_today:
            return
        if not (atr_val and baseline > 0):
            return

        # Check kill switch / cooldown for long portfolio on every entry-eligible bar
        triggered, ks_reason = self._long_kill_switch.check(self._long_portfolio, bar.timestamp)
        if triggered:
            logger.warning("KILL SWITCH: %s — long entries halted for session", ks_reason)
            self._sync_dash()
            return
        if self._long_portfolio.in_cooldown(bar.timestamp):
            return

        # --- Short entry (ETB-only, independent heat budget) ---
        if sym in self._etb_set and bar.close >= self._short_cfg.stage1_min_price:
            if baseline * 390 * bar.close >= self._short_cfg.min_avg_dollar_volume:
                can_enter, reason = self._short_portfolio.can_enter(sector="Unknown", now=bar.timestamp)
                if can_enter and self._short_validator.validate(bar, baseline):
                    self._enter_short(sym, bar, atr_val, baseline)
                    return
                elif not can_enter:
                    logger.debug("Short blocked for %s: %s", sym, reason)

        # --- Regime filter: check once per day, skip longs in SPY downtrend ---
        if self._regime["date"] != today:
            self._regime["uptrend"] = self._regime_filter.is_uptrend(today)
            self._regime["date"] = today
            if self._regime["uptrend"]:
                logger.info("REGIME %s: SPY uptrend — long entries enabled", today)
            else:
                logger.info("REGIME %s: SPY below 20-day MA — long entries blocked", today)
            self._sync_dash()

        # --- Long entry (independent heat budget) ---
        if self._regime["uptrend"] and bar.close >= self._long_cfg.stage1_min_price:
            if baseline * 390 * bar.close >= self._long_cfg.min_avg_dollar_volume:
                # stage2_min_dist_from_day_high_pct: entry must be ≥2% below intraday high.
                # Avoids chasing the absolute top of a spike — same check as the backtest simulator.
                day_high = self._day_highs.get(sym, bar.high)
                min_dist = self._long_cfg.stage2_min_dist_from_day_high_pct
                if min_dist > 0 and day_high > 0 and (day_high - bar.close) / day_high < min_dist:
                    return  # too close to the day's high — skip

                can_enter, reason = self._long_portfolio.can_enter(sector="Unknown", now=bar.timestamp)
                if can_enter and self._long_validator.validate(bar, baseline):
                    has_cat = self._news_filter.has_catalyst(sym, today)
                    if not has_cat:
                        # Tier-4 bypass: extreme confidence signal (4× mult) doesn't need
                        # external news — the volume + ROC combination IS the catalyst.
                        conf = self._long_validator.confidence_score(bar, baseline)
                        if self._long_cfg.confidence_multiplier(conf) >= 4.0:
                            logger.info("Tier-4 bypass for %s — entering without news catalyst", sym)
                            has_cat = True
                        else:
                            logger.debug("No catalyst for %s — skipping entry", sym)
                    if has_cat:
                        self._enter_long(sym, bar, atr_val, baseline)
                elif not can_enter:
                    logger.debug("Long blocked for %s: %s", sym, reason)

    def _enter_short(self, sym: str, bar: Bar, atr_val: float, baseline: float) -> None:
        size = compute_position_size(self._equity, atr_val, bar.close, self._short_cfg)
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
        )
        self._sync_dash()

    def _enter_long(self, sym: str, bar: Bar, atr_val: float, baseline: float) -> None:
        cfg = self._long_cfg
        size = compute_position_size(self._equity, atr_val, bar.close, cfg)
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
        )
        self._sync_dash()

    def _execute_close(self, sym: str, fill_price: float, reason: str, portfolio: PortfolioState) -> None:
        with self._close_lock:
            position = portfolio.remove_position(sym)
        if not position:
            return
        with self._open_syms_lock:
            self._open_syms.discard(sym)

        is_short = position.direction == "short"
        if is_short:
            pnl = round((position.entry_price - fill_price) * position.shares, 2)
        else:
            pnl = round((fill_price - position.entry_price) * position.shares, 2)
        self._sync_equity(self._equity + pnl)

        # Cancel broker stop before covering.
        # Returns False if Alpaca already filled the stop (broker covered/sold for us).
        need_to_cover = self._cancel_broker_stop(sym, position)

        closed_record: Optional[TradeRecord] = None
        if need_to_cover:
            try:
                order_id = broker.buy_to_cover(sym, position.shares) if is_short else broker.sell(sym, position.shares)
                self._state.remove_position(sym)
                action = "COVER" if is_short else "SELL"
                logger.info(
                    "%s %s x%d @ ~%.2f pnl=%.2f reason=%s equity=%.2f order=%s",
                    action, sym, position.shares, fill_price, pnl, reason,
                    self._equity, order_id,
                )
            except Exception as exc:
                # Restore position — neither the exit nor the stop order worked
                portfolio.add_position(position)
                self._sync_equity(self._equity - pnl)
                with self._open_syms_lock:
                    self._open_syms.add(sym)
                logger.error("Exit order failed for %s: %s — position restored", sym, exc)
                return
        else:
            self._state.remove_position(sym)
            action = "COVERED" if is_short else "SOLD"
            logger.info(
                "%s by broker stop %s x%d pnl≈%.2f reason=%s equity≈%.2f",
                action, sym, position.shares, pnl, reason, self._equity,
            )

        # Update consecutive loss counter on the relevant portfolio
        if pnl < 0:
            portfolio.consecutive_losses += 1
        else:
            portfolio.consecutive_losses = 0

        record = self._open_records.pop(sym, None)
        if record is not None:
            record.exit_time = datetime.now(_ET)
            record.exit_price = fill_price
            record.pnl = pnl
            record.exit_reason = reason
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
        watchlist: Set[str] = set()
        for m in movers:
            sym = m.get("symbol", "")
            pct = m.get("percent_change", 0.0)
            price = m.get("price", 0.0)
            if pct < _MIN_MOVER_PCT:
                continue
            if price < min_price:
                continue
            watchlist.add(sym)
        return watchlist

    def _refresh_watchlist_loop(self) -> None:
        """Periodically fetch movers and queue sub/unsub requests. Daemon thread."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=_WATCHLIST_REFRESH_SEC)
            if self._stop_event.is_set():
                break
            try:
                movers = broker_ibkr.get_movers(
                    self._ibkr_host, self._ibkr_port,
                    self._ibkr_scanner_client_id, top_n=200,
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

    # ------------------------------------------------------------------
    # Status log — daemon thread
    # ------------------------------------------------------------------

    def _status_log_loop(self) -> None:
        """Log equity, open positions, and watchlist size every minute. Daemon thread."""
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

    # ------------------------------------------------------------------
    # Baseline volume helpers
    # ------------------------------------------------------------------

    def _fetch_baseline_volumes(self, symbols: List[str]) -> None:
        import yfinance as yf
        import pandas as pd

        today = date.today()
        start = (today - timedelta(days=35)).isoformat()
        end = today.isoformat()
        try:
            raw = yf.download(
                tickers=symbols, interval="1d", start=start, end=end,
                progress=False, auto_adjust=False, group_by="ticker", threads=False,
            )
        except Exception as exc:
            logger.warning("Baseline volume fetch failed for %d symbols: %s", len(symbols), exc)
            return

        updates: Dict[str, float] = {}
        for sym in symbols:
            try:
                df = raw if len(symbols) == 1 else (
                    raw[sym] if sym in raw.columns.get_level_values(0) else pd.DataFrame()
                )
                if df.empty:
                    continue
                avg_vol = float(df["Volume"].tail(20).mean())
                updates[sym] = avg_vol / 390
            except Exception:
                pass
        if updates:
            with self._baseline_vols_lock:
                self._baseline_vols.update(updates)
        logger.info("Baseline volumes loaded for %d/%d symbols", len(updates), len(symbols))

    def load_baseline_volumes(self, symbols: List[str]) -> None:
        logger.info("Fetching baseline volumes for %d initial symbols...", len(symbols))
        self._fetch_baseline_volumes(symbols)

    def _flush_bar_cache(self) -> None:
        """Write accumulated IBKR bars to the backtest cache directory.

        Skips symbols whose file already exists (Alpaca historical data takes
        precedence for past dates; this only fills in today's bars going forward).
        """
        today = date.today()
        written = skipped = 0
        for sym, bars in self._live_bar_cache.items():
            if not bars:
                continue
            cache_path = self._bar_cache_dir / f"{sym}_{today}.json"
            if cache_path.exists():
                skipped += 1
                continue
            cache_path.write_text(json.dumps(bars))
            written += 1
        logger.info(
            "IBKR bar cache flushed: %d files written, %d skipped (already cached) for %s",
            written, skipped, today,
        )
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
        """Submit market close orders for all open positions. Called from EOD timer thread.

        Uses _close_lock to atomically claim each position before submitting the
        broker order, preventing a double-close race with _on_bar/_execute_close
        (which runs on the asyncio event loop thread for halted stocks that resume
        near 15:55 with a late bar).
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
            with self._close_lock:
                pos = self._short_portfolio.positions.get(sym)
                if pos is None:
                    continue
                self._short_portfolio.remove_position(sym)
            try:
                if pos.stop_order_id:
                    try:
                        broker.cancel_order(pos.stop_order_id)
                    except Exception:
                        pass
                broker.buy_to_cover(sym, pos.shares)
                with self._open_syms_lock:
                    self._open_syms.discard(sym)
                self._state.remove_position(sym)
                logger.info("EOD timer: COVERED short %s x%d", sym, pos.shares)
            except Exception as exc:
                logger.error("EOD timer: failed to cover %s: %s — check account manually", sym, exc)
        for sym in long_syms:
            with self._close_lock:
                pos = self._long_portfolio.positions.get(sym)
                if pos is None:
                    continue
                self._long_portfolio.remove_position(sym)
            try:
                if pos.stop_order_id:
                    try:
                        broker.cancel_order(pos.stop_order_id)
                    except Exception:
                        pass
                broker.sell(sym, pos.shares)
                with self._open_syms_lock:
                    self._open_syms.discard(sym)
                self._state.remove_position(sym)
                logger.info("EOD timer: SOLD long %s x%d", sym, pos.shares)
            except Exception as exc:
                logger.error("EOD timer: failed to sell %s: %s — check account manually", sym, exc)

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
                self._ibkr_scanner_client_id, top_n=200,
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

        self._stream = BarStream(self._ibkr_host, self._ibkr_port, self._ibkr_client_id, list(all_initial))
        self._stream.set_handler(self._on_bar)

        if self._dash is not None:
            from bot.dashboard.server import start_server
            self._sync_dash()
            start_server(self._dash)

        for name, target in [
            ("watchlist-refresh", self._refresh_watchlist_loop),
            ("status-log", self._status_log_loop),
            ("etb-refresh", self._etb_refresh_loop),
            ("eod-timer", self._eod_timer_loop),
        ]:
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
