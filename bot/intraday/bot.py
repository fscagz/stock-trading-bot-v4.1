"""
V3 Intraday Bot — main orchestrator.

Trade lifecycle per incoming 1-min bar:
  1. Update indicators (VWAP, ATR, RSI, EMA)
  2. Generate technical signals for this bar
  3. For each symbol with signals:
     a. Portfolio checks (heat, sector, kill switch, cooldown, macro blackout)
     b. Sentiment gate (block if news contradicts direction)
     c. Correlation cap (block if too correlated with existing position)
     d. ATR-based position sizing (ML-adjusted if model loaded)
     e. Submit aggressive limit entry order
     f. On fill: submit OCO bracket; log entry; update PortfolioState
  4. Bracket fills handled via TradingStream callbacks: log exit, update equity
  5. Forced EOD close at eod_close: cancel brackets, submit market orders, update equity
"""
from __future__ import annotations
import dataclasses
import logging
import os
import threading
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

from bot.intraday.config import IntradayConfig
from bot.intraday.data.event_calendar import EventCalendar
from bot.intraday.data.stream import BarStream
from bot.intraday.data.news_stream import NewsStream
from bot.intraday.data.trade_stream import TradeStream
from bot.intraday.execution.brackets import BracketManager
from bot.intraday.execution.orders import OrderManager
from bot.intraday.execution.trade_log import TradeLogger
from bot.intraday.indicators.atr import ATRIndicator
from bot.intraday.indicators.ema import EMAIndicator
from bot.intraday.indicators.rsi import RSIIndicator
from bot.intraday.indicators.vwap import VWAPIndicator
from bot.intraday.risk.kill_switch import KillSwitch
from bot.intraday.risk.portfolio import PortfolioState
from bot.intraday.risk.regime import Regime, RegimeDetector
from bot.intraday.risk.sizing import compute_position_size
from bot.intraday.ml.scorer import MLScorer
from bot.intraday.signals.aggregator import SignalAggregator
from bot.intraday.signals.event import check_earnings_signal
from bot.intraday.signals.sentiment import SentimentAggregator, SentimentScorer
from bot.intraday.signals.technical import (
    check_breakout,
    check_ma_crossover,
    check_momentum_burst,
    check_rsi_extreme,
    check_vwap_continuation,
)
from bot.intraday.types import Bar, Position, TradeRecord

logger = logging.getLogger(__name__)


class IntradayBot:
    """
    Main V3 intraday bot.

    Usage:
        bot = IntradayBot(config, broker, symbols)
        bot.initialize_session(equity=25_000.0, regime=Regime.TRENDING_BULL)
        bot.start()  # blocks
    """

    def __init__(
        self,
        config: IntradayConfig,
        broker,
        symbols: List[str],
        trade_log_path: str = "bot/trade_log.csv",
        sector_map: Optional[Dict[str, str]] = None,
        model_path: Optional[str] = None,
        correlation_map: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self._cfg = config
        self._broker = broker
        self._symbols = symbols
        self._sector_map = sector_map or {}
        self._correlation_map: Dict[str, Dict[str, float]] = correlation_map or {}

        api_key = os.environ["APCA_API_KEY_ID"]
        secret_key = os.environ["APCA_API_SECRET_KEY"]

        self._bar_stream = BarStream(api_key, secret_key, symbols)
        self._news_stream = NewsStream(api_key, secret_key, symbols)
        self._trade_stream = TradeStream(api_key, secret_key)
        self._bar_stream.set_handler(self._on_bar)
        self._news_stream.set_handler(self._on_news)
        self._trade_stream.set_handler(self._on_trade_update)

        # Indicators (one instance per indicator type; keyed internally by symbol)
        self._vwap = VWAPIndicator()
        self._atr = ATRIndicator(period=14)
        self._rsi = RSIIndicator(period=14)
        self._ema9 = EMAIndicator(period=9)
        self._ema21 = EMAIndicator(period=21)
        self._ema50 = EMAIndicator(period=50)

        # Signals
        self._aggregator = SignalAggregator(config)
        self._sentiment_scorer = SentimentScorer()
        self._sentiment_agg = SentimentAggregator(config.sentiment_half_life_minutes)

        # Data
        self._event_calendar = EventCalendar()
        self._regime_detector = RegimeDetector(config)

        # Risk
        self._kill_switch = KillSwitch(config)

        # Execution
        self._order_manager = OrderManager(broker, config)
        self._bracket_manager = BracketManager(broker, config)
        self._trade_logger = TradeLogger(trade_log_path)

        # ML scorer — optional; None means rule-based mode (full size always)
        self._ml_scorer: Optional[MLScorer] = None
        if model_path and os.path.exists(model_path):
            try:
                self._ml_scorer = MLScorer(model_path, config)
                logger.info("ML scorer loaded from %s", model_path)
            except Exception as exc:
                logger.warning("Failed to load ML scorer from %s: %s", model_path, exc)

        # Thread lock — guards portfolio state, positions, and tracking dicts
        # accessed from bar stream thread AND trade update stream thread.
        self._lock = threading.Lock()

        # State (set by initialize_session before start)
        self._portfolio: Optional[PortfolioState] = None
        self._current_regime: Regime = Regime.RANGE_BOUND

        # Per-symbol rolling stats
        self._avg_volume: Dict[str, float] = {}
        self._bar_count: Dict[str, int] = {}
        self._session_trend: Dict[str, str] = {}
        self._last_price: Dict[str, float] = {}

        # Prior session high/low — populated from _current_session_* at daily reset
        self._prior_session_high: Dict[str, float] = {}
        self._prior_session_low: Dict[str, float] = {}
        # Current-session running high/low — updated every bar, reset at daily reset
        self._current_session_high: Dict[str, float] = {}
        self._current_session_low: Dict[str, float] = {}

        # ORB (Opening Range Breakout) state — per symbol, reset daily
        _orb_total = 9 * 60 + 30 + config.opening_range_minutes  # minutes since midnight
        self._orb_end_str: str = f"{_orb_total // 60:02d}:{_orb_total % 60:02d}"
        self._orb_high: Dict[str, float] = {}
        self._orb_low: Dict[str, float] = {}
        self._orb_confirmed: Dict[str, bool] = {}
        self._prior_close: Dict[str, float] = {}
        self._last_session_date: Optional[date] = None

        # Bracket order tracking — needed to cancel open legs at EOD and to
        # correlate TradingStream fill events back to positions.
        # _position_orders: ticker → (stop_order_id, target_order_id)
        self._position_orders: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        # _order_to_ticker: order_id → ticker (O(1) lookup in _on_trade_update)
        self._order_to_ticker: Dict[str, str] = {}
        # _order_side: order_id → "stop" | "target"
        self._order_side: Dict[str, str] = {}

    def initialize_session(self, equity: float, regime: Regime) -> None:
        self._portfolio = PortfolioState(equity=equity, config=self._cfg)
        self._current_regime = regime
        self._event_calendar.load_earnings(self._symbols)
        logger.info("Session initialized. Equity=%.2f Regime=%s", equity, regime.value)

    def _et_time_str(self, ts: datetime) -> str:
        """Return HH:MM string in Eastern Time (simplified: UTC-4 for EDT)."""
        et_hour = (ts.hour - 4) % 24
        return f"{et_hour:02d}:{ts.minute:02d}"

    def _on_bar(self, bar: Bar) -> None:
        if self._portfolio is None:
            return

        sym = bar.symbol
        et_str = self._et_time_str(bar.timestamp)

        with self._lock:
            # Force-close all positions at EOD
            if et_str >= self._cfg.eod_close:
                self._close_all_eod(bar.timestamp)
                return

            # Daily reset when the trading date changes
            bar_date = bar.timestamp.date()
            if bar_date != self._last_session_date:
                # Copy current-session high/low to prior before clearing
                for s, h in self._current_session_high.items():
                    self._prior_session_high[s] = h
                for s, lo in self._current_session_low.items():
                    self._prior_session_low[s] = lo
                self._current_session_high.clear()
                self._current_session_low.clear()

                self._last_session_date = bar_date
                self._orb_high.clear()
                self._orb_low.clear()
                self._orb_confirmed.clear()
                self._bar_count.clear()
                self._avg_volume.clear()
                self._session_trend.clear()

            # Always track last price
            self._last_price[sym] = bar.close

            # Track prior close from any pre-session bar
            if et_str < self._cfg.session_start:
                self._prior_close[sym] = bar.close

            # Track ORB window: market open (09:30) through orb_end
            if "09:30" <= et_str < self._orb_end_str:
                self._orb_high[sym] = max(self._orb_high.get(sym, 0.0), bar.high)
                self._orb_low[sym] = min(self._orb_low.get(sym, float("inf")), bar.low)
            elif et_str >= self._orb_end_str and not self._orb_confirmed.get(sym, False):
                self._orb_confirmed[sym] = True

            # Update indicators (runs on every bar including pre-session for warmup)
            vwap = self._vwap.update(bar)
            atr = self._atr.update(bar)
            rsi = self._rsi.update(bar)
            ema9 = self._ema9.update(bar)
            ema21 = self._ema21.update(bar)
            ema50 = self._ema50.update(bar)

            # Update rolling avg volume
            bc = self._bar_count.get(sym, 0) + 1
            self._bar_count[sym] = bc
            prev_avg = self._avg_volume.get(sym, float(bar.volume))
            self._avg_volume[sym] = prev_avg + (bar.volume - prev_avg) / min(bc, 20)

            # Update current-session high/low (only during trading hours)
            if et_str >= "09:30":
                prev_high = self._current_session_high.get(sym, bar.high)
                self._current_session_high[sym] = max(prev_high, bar.high)
                prev_low = self._current_session_low.get(sym, bar.low)
                self._current_session_low[sym] = min(prev_low, bar.low)

            # Skip before session window or before indicators warm up
            if et_str < self._cfg.session_start:
                return
            if atr is None or rsi is None or ema9 is None or ema21 is None or ema50 is None:
                return
            # No new entries after session_end
            if et_str >= self._cfg.session_end:
                return

            # Derive session trend from EMA9 vs EMA50
            trend = "up" if ema9 > ema50 else "down"
            self._session_trend[sym] = trend

            avg_vol = self._avg_volume[sym]
            prior_high = self._prior_session_high.get(sym, bar.high)
            prior_low = self._prior_session_low.get(sym, bar.low)

            # Generate technical signals
            for check_fn, kwargs in [
                (check_vwap_continuation, {"vwap": vwap, "avg_volume_20": avg_vol, "session_trend": trend}),
                (check_momentum_burst, {"avg_volume_20": avg_vol, "session_trend": trend}),
                (check_rsi_extreme, {"rsi": rsi, "session_trend": trend}),
                (check_breakout, {"prior_session_high": prior_high, "prior_session_low": prior_low,
                                  "avg_volume_20": avg_vol}),
            ]:
                sig = check_fn(bar=bar, config=self._cfg, **kwargs)
                if sig:
                    self._aggregator.add(sig)

            ema50_slope = "up" if trend == "up" else "down"
            ma_sig = check_ma_crossover(bar, ema9, ema21, ema50_slope, avg_vol, self._cfg)
            if ma_sig:
                self._aggregator.add(ma_sig)

            # Generate earnings signal if ORB is confirmed and ticker has earnings today
            if self._orb_confirmed.get(sym, False):
                event = self._event_calendar.get_earnings(bar.timestamp.date(), sym)
                if event is not None:
                    earn_sig = check_earnings_signal(
                        bar=bar,
                        event=event,
                        prior_close=self._prior_close.get(sym, bar.close),
                        opening_range_high=self._orb_high.get(sym),
                        opening_range_low=self._orb_low.get(sym),
                        orb_confirmed=True,
                        config=self._cfg,
                    )
                    if earn_sig:
                        self._aggregator.add(earn_sig)

            # Attempt entry
            self._try_entry(bar, atr)

            # Clear bar-level signals
            self._aggregator.clear(sym)

    def _on_news(self, ticker: str, headline: str, timestamp: datetime) -> None:
        score = self._sentiment_scorer.score(headline)
        self._sentiment_agg.add_score(ticker, score, timestamp)

    def _on_trade_update(self, data) -> None:
        """Handle Alpaca TradingStream fill events (runs in trade-stream thread)."""
        try:
            event_type = getattr(data, "event", None)
            if event_type != "fill":
                return

            order = getattr(data, "order", None)
            if order is None:
                return

            order_id = str(getattr(order, "id", ""))
            fill_price_raw = getattr(order, "filled_avg_price", None)
            if not order_id or fill_price_raw is None:
                return

            exit_price = float(fill_price_raw)

            with self._lock:
                ticker = self._order_to_ticker.get(order_id)
                if ticker is None:
                    return  # not a bracket order we're tracking

                assert self._portfolio is not None
                pos = self._portfolio.positions.get(ticker)
                if pos is None:
                    return

                side = self._order_side.get(order_id, "unknown")
                exit_reason = "stop_hit" if side == "stop" else "target_hit"

                # Cancel the remaining bracket leg
                stop_id, target_id = self._position_orders.get(ticker, (None, None))
                other_id = target_id if side == "stop" else stop_id
                if other_id:
                    try:
                        self._broker.cancel_order_by_id(other_id)
                    except Exception as exc:
                        logger.warning("Could not cancel bracket leg %s: %s", other_id, exc)

                # PnL and equity update
                pnl = self._calculate_pnl(pos, exit_price)
                self._portfolio.equity += pnl

                if pnl < 0:
                    self._portfolio.consecutive_losses += 1
                else:
                    self._portfolio.consecutive_losses = 0

                # Log exit
                now_utc = datetime.now(timezone.utc)
                self._trade_logger.log_exit(
                    ticker=ticker,
                    entry_time=pos.entry_time,
                    exit_price=exit_price,
                    exit_time=now_utc,
                    exit_reason=exit_reason,
                    actual_slippage_pct=self._cfg.expected_exit_slippage_pct,
                )
                logger.info(
                    "EXIT %s %s %d@%.2f pnl=%.2f reason=%s",
                    pos.direction.upper(), ticker, pos.shares, exit_price, pnl, exit_reason,
                )

                # Clean up
                self._portfolio.remove_position(ticker)
                self._position_orders.pop(ticker, None)
                for oid in [stop_id, target_id]:
                    if oid:
                        self._order_to_ticker.pop(oid, None)
                        self._order_side.pop(oid, None)

        except Exception as exc:
            logger.error("Error in _on_trade_update: %s", exc)

    def _calculate_pnl(self, pos: "Position", exit_price: float) -> float:
        if pos.direction == "long":
            return (exit_price - pos.entry_price) * pos.shares
        else:
            return (pos.entry_price - exit_price) * pos.shares

    def _build_ml_features(
        self,
        portfolio: "PortfolioState",
        signal_types: List[str],
        regime: Regime,
    ) -> dict:
        """Build ML feature dict matching ml/features.py FEATURE_COLS."""
        from bot.intraday.ml.features import SIGNAL_KEYWORDS, REGIME_VALUES
        signals_str = " ".join(signal_types)
        features: dict = {"portfolio_heat_at_entry": portfolio.portfolio_heat_pct}
        for col, kw in SIGNAL_KEYWORDS.items():
            features[col] = int(kw in signals_str)
        for r in REGIME_VALUES:
            features[f"regime_{r}"] = int(regime.value == r)
        return features

    def _try_entry(self, bar: Bar, atr: float) -> None:
        """Attempt to enter a position. Must be called with self._lock held."""
        assert self._portfolio is not None

        ticker = bar.symbol
        signals = self._aggregator.get_signals(ticker)
        if not signals or self._aggregator.has_conflict(ticker):
            return

        direction = signals[0].direction
        now = bar.timestamp
        sector = self._sector_map.get(ticker, "Unknown")

        # Portfolio and risk checks
        if self._current_regime == Regime.CRASH:
            return
        if self._current_regime == Regime.HIGH_VOL and len(self._portfolio.positions) >= 2:
            return

        allowed, reason = self._portfolio.can_enter(sector, now)
        if not allowed:
            logger.debug("Entry blocked for %s: %s", ticker, reason)
            return

        if self._event_calendar.is_macro_blackout(now):
            logger.debug("Entry blocked for %s: macro blackout", ticker)
            return

        triggered, kill_reason = self._kill_switch.check(self._portfolio, now)
        if triggered:
            logger.warning("Kill switch: %s", kill_reason)
            return

        # Sentiment gate: block entry if news strongly contradicts direction
        sentiment = self._sentiment_agg.aggregate(ticker, now)
        if direction == "long" and sentiment < -self._cfg.sentiment_threshold:
            logger.debug("Entry blocked for %s: bearish sentiment %.2f", ticker, sentiment)
            return
        if direction == "short" and sentiment > self._cfg.sentiment_threshold:
            logger.debug("Entry blocked for %s: bullish sentiment %.2f", ticker, sentiment)
            return

        # Correlation cap: skip if too correlated with any existing position
        corr_row = self._correlation_map.get(ticker, {})
        for existing in self._portfolio.positions:
            corr = abs(corr_row.get(existing, 0.0))
            if corr >= self._cfg.max_position_correlation:
                logger.debug(
                    "Entry blocked for %s: corr=%.2f with %s >= %.2f",
                    ticker, corr, existing, self._cfg.max_position_correlation,
                )
                return

        # Size
        size = compute_position_size(
            equity=self._portfolio.equity,
            atr=atr,
            entry_price=bar.close,
            config=self._cfg,
        )
        if size.shares == 0:
            return

        # ML scoring — apply position size multiplier if scorer is loaded
        ml_probability: Optional[float] = None
        if self._ml_scorer is not None:
            signal_types = [s.signal_type for s in signals]
            features = self._build_ml_features(self._portfolio, signal_types, self._current_regime)
            ml_probability = self._ml_scorer.score(features)
            multiplier = self._ml_scorer.position_size_multiplier(ml_probability)
            if multiplier == 0.0:
                logger.debug("ML scorer filtered trade on %s (prob=%.3f)", ticker,
                             ml_probability or 0.0)
                return
            size = dataclasses.replace(size, shares=max(1, int(size.shares * multiplier)))

        # Submit entry
        fill = self._order_manager.submit_entry(ticker, direction, size.shares, bar.close)
        if fill is None or not fill.complete:
            return

        entry_price = fill.fill_price
        stop_price = size.long_stop(entry_price) if direction == "long" else size.short_stop(entry_price)
        target_price = size.long_target(entry_price) if direction == "long" else size.short_target(entry_price)

        # Submit OCO bracket and record order IDs for callback tracking
        stop_id, target_id = self._bracket_manager.submit_bracket(
            ticker, direction, fill.filled_shares, stop_price, target_price
        )
        self._position_orders[ticker] = (stop_id, target_id)
        for oid, side_label in [(stop_id, "stop"), (target_id, "target")]:
            if oid:
                self._order_to_ticker[oid] = ticker
                self._order_side[oid] = side_label

        # Update portfolio state
        open_risk = fill.filled_shares * size.stop_distance
        position = Position(
            ticker=ticker, direction=direction,
            shares=fill.filled_shares, entry_price=entry_price,
            stop_price=stop_price, target_price=target_price,
            entry_time=now, atr_at_entry=atr,
            signals=[s.signal_type for s in signals],
            sector=sector, open_risk=open_risk,
        )
        self._portfolio.add_position(position)
        self._portfolio.session_slippage_expected += self._cfg.expected_entry_slippage_pct
        self._portfolio.session_slippage_actual += fill.slippage_pct

        # Log entry
        record = TradeRecord(
            ticker=ticker, direction=direction,
            entry_time=now, entry_price=entry_price,
            shares=fill.filled_shares,
            stop_price=stop_price, target_price=target_price,
            signals=[s.signal_type for s in signals],
            sector=sector, regime=self._current_regime.value,
            portfolio_heat_at_entry=self._portfolio.portfolio_heat_pct,
            expected_slippage_pct=self._cfg.expected_entry_slippage_pct,
            ml_score=ml_probability,
        )
        self._trade_logger.log_entry(record)
        logger.info(
            "ENTRY %s %s %d@%.2f stop=%.2f target=%.2f",
            direction.upper(), ticker, fill.filled_shares,
            entry_price, stop_price, target_price,
        )

    def _close_all_eod(self, now: datetime) -> None:
        """Force-close all open positions at EOD. Must be called with self._lock held."""
        assert self._portfolio is not None
        for ticker, pos in list(self._portfolio.positions.items()):
            try:
                # Cancel open bracket orders before submitting market close
                stop_id, target_id = self._position_orders.get(ticker, (None, None))
                for oid in [stop_id, target_id]:
                    if oid:
                        try:
                            self._broker.cancel_order_by_id(oid)
                        except Exception as exc:
                            logger.warning("Could not cancel bracket %s at EOD: %s", oid, exc)

                # Market close order
                side = "sell" if pos.direction == "long" else "buy"
                self._broker.submit_order(
                    symbol=ticker, qty=pos.shares, side=side,
                    type="market", time_in_force="day",
                )

                # Use last known price for PnL (more accurate than entry_price placeholder)
                exit_price = self._last_price.get(ticker, pos.entry_price)
                pnl = self._calculate_pnl(pos, exit_price)
                self._portfolio.equity += pnl

                if pnl < 0:
                    self._portfolio.consecutive_losses += 1
                else:
                    self._portfolio.consecutive_losses = 0

                self._trade_logger.log_exit(
                    ticker=ticker,
                    entry_time=pos.entry_time,
                    exit_price=exit_price,
                    exit_time=now,
                    exit_reason="eod_close",
                    actual_slippage_pct=self._cfg.expected_exit_slippage_pct,
                )
                self._portfolio.remove_position(ticker)

                # Clean up bracket tracking
                self._position_orders.pop(ticker, None)
                for oid in [stop_id, target_id]:
                    if oid:
                        self._order_to_ticker.pop(oid, None)
                        self._order_side.pop(oid, None)

                logger.info("EOD CLOSE %s @%.2f pnl=%.2f", ticker, exit_price, pnl)
            except Exception as exc:
                logger.error("EOD close failed for %s: %s", ticker, exc)

    def start(self) -> None:
        """Start all three streams. Blocks on the bar stream (primary thread)."""
        news_thread = threading.Thread(target=self._news_stream.run, daemon=True)
        news_thread.start()

        trade_thread = threading.Thread(target=self._trade_stream.run, daemon=True)
        trade_thread.start()

        logger.info("Bot started. Streaming %d symbols.", len(self._symbols))
        self._bar_stream.run()
