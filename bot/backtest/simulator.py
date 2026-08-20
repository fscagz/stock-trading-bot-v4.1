from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime
from typing import Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

from bot.backtest.news_filter import NewsFilter
from bot.config import V4Config
from bot.intraday.indicators.atr import ATRIndicator
from bot.intraday.indicators.vwap import VWAPIndicator
from bot.intraday.risk.kill_switch import KillSwitch
from bot.intraday.risk.portfolio import PortfolioState
from bot.intraday.risk.sizing import compute_position_size
from bot.intraday.types import Bar, Position, TradeRecord
from bot.momentum.validator import MomentumValidator
from bot.positions.manager import PositionManager

_ET = ZoneInfo("America/New_York")
_EOD_HOUR = 15
_EOD_MINUTE = 55   # matches LiveRunner _EOD_FORCE_CLOSE = dtime(15, 55)
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    trades: List[TradeRecord]
    equity_curve: List[Tuple[datetime, float]]


class Simulator:
    def __init__(
        self,
        config: V4Config,
        initial_equity: float,
        slippage_pct: float = 0.001,
        overnight_holds: bool = True,
        market_order_fill: bool = False,
        news_filter: Optional[NewsFilter] = None,
        news_mode: str = "ignore",
        require_above_vwap_at_entry: bool = False,
        news_relvol_bypass: float = 0.0,
        news_tier_bypass: int = 0,
        max_position_dv_pct: float = 0.0,
        min_entry_tier: int = 0,
        stage2_min_dist_from_day_high_pct: float = 0.0,
        pullback_entry_atr: float = 0.0,
        pullback_entry_ttl_bars: int = 30,
        pullback_chase_tier: int = 0,
        pullback_target_atr: float = 0.0,
        tier4_only: bool = False,
        max_range_position_pct: float = 0.0,
        max_entry_bar_range_pct: float = 0.0,
        gap_hold_entry: bool = False,
        gap_hold_min_pct: float = 0.05,
        gap_hold_bars: int = 15,
        gap_hold_tolerance: float = 0.02,
        # Short-side parameters
        short_config: Optional[V4Config] = None,
        etb_set: Optional[Set[str]] = None,
        # Realistic fill model: cap fills at a fraction of the fill bar's volume
        # and charge participation-proportional impact on top of slippage_pct.
        # 0.0 disables both (legacy behavior: any size fills at quoted price).
        max_bar_participation: float = 0.0,
        impact_slippage_coeff: float = 0.0,
    ) -> None:
        self._config = config
        self._initial_equity = initial_equity
        self._slippage_pct = slippage_pct
        self._overnight_holds = overnight_holds
        self._market_order_fill = market_order_fill
        self._news_filter = news_filter
        self._news_mode = news_mode
        self._require_above_vwap_at_entry = require_above_vwap_at_entry
        self._news_relvol_bypass = news_relvol_bypass
        self._news_tier_bypass = news_tier_bypass
        self._min_entry_tier = min_entry_tier
        self._max_position_dv_pct = max_position_dv_pct
        self._stage2_min_dist_from_day_high_pct = stage2_min_dist_from_day_high_pct
        # Pullback entry: instead of chasing the signal bar at market, arm a limit
        # order pullback_entry_atr × ATR below the signal close, valid for
        # pullback_entry_ttl_bars of that symbol's bars, then expire unfilled.
        # Mirrors a live limit order with a TTL cancel.
        self._pullback_entry_atr = pullback_entry_atr
        self._pullback_entry_ttl_bars = pullback_entry_ttl_bars
        # Hybrid mode: signals at/above this tier skip the pullback limit and
        # enter at market immediately (extreme signals don't wait for a flush).
        self._pullback_chase_tier = pullback_chase_tier
        # If > 0, pullback fills use this target multiple instead of
        # config.target_atr_multiple (which chase entries keep).
        self._pullback_target_atr = pullback_target_atr
        # When True, skip signals below pullback_chase_tier entirely (no pullback arm).
        self._tier4_only = tier4_only
        self._max_range_position_pct = max_range_position_pct
        self._max_entry_bar_range_pct = max_entry_bar_range_pct
        # Gap-and-hold: enter after a stock gaps up at open and holds the gap
        # for gap_hold_bars bars without retracing more than gap_hold_tolerance.
        self._gap_hold_entry = gap_hold_entry
        self._gap_hold_min_pct = gap_hold_min_pct
        self._gap_hold_bars = gap_hold_bars
        self._gap_hold_tolerance = gap_hold_tolerance
        # Short-side state
        self._short_config = short_config
        self._etb_set: Set[str] = etb_set if etb_set is not None else set()
        self._max_bar_participation = max_bar_participation
        self._impact_slippage_coeff = impact_slippage_coeff

    def _liquidity_fill(self, shares: int, bar: Bar) -> Tuple[int, float]:
        """Cap shares to a fraction of the fill bar's volume; return (shares, slippage).

        Slippage = slippage_pct + impact_coeff × participation, where participation
        is the fraction of the bar's traded volume we consume (capped at 1.0).
        With max_bar_participation=0 this is a no-op returning legacy slippage.
        """
        if self._max_bar_participation <= 0:
            return shares, self._slippage_pct
        if bar.volume <= 0:
            return 0, self._slippage_pct
        cap = int(bar.volume * self._max_bar_participation)
        shares = min(shares, cap)
        participation = min(1.0, shares / bar.volume)
        return shares, self._slippage_pct + self._impact_slippage_coeff * participation

    def _exit_slippage(self, position: Position, exit_bar: Optional[Bar]) -> float:
        """Adverse exit slippage under the realistic fill model (0.0 when disabled)."""
        if self._max_bar_participation <= 0 or exit_bar is None:
            return 0.0
        if exit_bar.volume <= 0:
            return self._slippage_pct + self._impact_slippage_coeff
        participation = min(1.0, position.shares / exit_bar.volume)
        return self._slippage_pct + self._impact_slippage_coeff * participation

    def run_day(
        self,
        trade_date: date,
        bars_by_symbol: Dict[str, List[Bar]],
        baseline_volumes: Dict[str, float],
        prev_closes: Optional[Dict[str, float]] = None,
    ) -> BacktestResult:
        atr_indicator = ATRIndicator(period=14)
        vwap_indicator = VWAPIndicator()
        validator = MomentumValidator(self._config)
        manager = PositionManager(self._config)
        kill_switch = KillSwitch(self._config)
        portfolio = PortfolioState(equity=self._initial_equity, config=self._config)
        last_bar: Dict[str, Bar] = {}
        prev_bar: Dict[str, Bar] = {}
        day_highs: Dict[str, float] = {}

        # Short-side parallel state (only allocated when short_config is set)
        short_portfolio: Optional[PortfolioState] = None
        short_open_records: Dict[str, TradeRecord] = {}
        short_traded_today: Set[str] = set()
        short_pos_peak: Dict[str, float] = {}
        short_pos_trough: Dict[str, float] = {}
        if self._short_config is not None:
            short_portfolio = PortfolioState(
                equity=self._initial_equity, config=self._short_config
            )

        # Pre-compute catalyst status for all symbols (avoids per-bar API calls).
        # When bypass params are set, keep all symbols and gate at entry time;
        # otherwise filter the symbol list here for efficiency.
        catalyst_status: Dict[str, bool] = {}
        _news_active = self._news_filter is not None and self._news_mode == "require"
        _has_bypass   = self._news_relvol_bypass > 0 or self._news_tier_bypass > 0
        if _news_active:
            for sym in bars_by_symbol:
                catalyst_status[sym] = self._news_filter.has_catalyst(sym, trade_date)  # type: ignore[union-attr]
            if not _has_bypass:
                bars_by_symbol = {s: b for s, b in bars_by_symbol.items() if catalyst_status[s]}
        if self._news_filter is not None and self._news_mode == "exclude":
            bars_by_symbol = {
                s: b for s, b in bars_by_symbol.items()
                if not self._news_filter.has_catalyst(s, trade_date)
            }

        merged: List[Bar] = sorted(
            (b for bars in bars_by_symbol.values() for b in bars),
            key=lambda b: b.timestamp,
        )

        open_records: Dict[str, TradeRecord] = {}
        closed_trades: List[TradeRecord] = []
        pending_entries: Dict[str, dict] = {}  # symbol → {atr_val, size}
        armed_limits: Dict[str, dict] = {}  # symbol → {limit, atr, mult, ttl}
        traded_today: set = set()  # symbols that already had an entry today
        equity_curve: List[Tuple[datetime, float]] = []
        day_lows: Dict[str, float] = {}
        day_opens: Dict[str, float] = {}
        gap_tracking: Dict[str, int] = {}  # sym → bars elapsed since gap detected
        gap_confirmed: set = set()         # symbols that completed the hold period
        pos_peak: Dict[str, float] = {}   # sym → highest high seen while position open
        pos_trough: Dict[str, float] = {} # sym → lowest low seen while position open

        for bar in merged:
            sym = bar.symbol
            baseline = baseline_volumes.get(sym, 0.0)
            bar_et = bar.timestamp.astimezone(_ET)
            prev_bar[sym] = last_bar.get(sym)  # capture BEFORE overwrite
            last_bar[sym] = bar
            day_highs[sym] = max(day_highs.get(sym, 0.0), bar.high)
            day_lows[sym] = min(day_lows[sym], bar.low) if sym in day_lows else bar.low

            # Gap-and-hold tracking: detect gap on first bar, count hold bars
            if self._gap_hold_entry and prev_closes is not None:
                if sym not in day_opens:
                    day_opens[sym] = bar.open
                    p_close = prev_closes.get(sym, 0.0)
                    if p_close > 0 and (bar.open - p_close) / p_close >= self._gap_hold_min_pct:
                        gap_tracking[sym] = 0
                if sym in gap_tracking:
                    d_open = day_opens.get(sym, bar.open)
                    if d_open > 0 and bar.close < d_open * (1 - self._gap_hold_tolerance):
                        del gap_tracking[sym]  # gap broke — abort
                    else:
                        gap_tracking[sym] += 1
                        if gap_tracking[sym] >= self._gap_hold_bars:
                            gap_confirmed.add(sym)
                            del gap_tracking[sym]
            if sym in portfolio.positions:
                pos_peak[sym] = max(pos_peak.get(sym, bar.high), bar.high)
                pos_trough[sym] = min(pos_trough.get(sym, bar.low), bar.low)

            if short_portfolio is not None and sym in short_portfolio.positions:
                short_pos_peak[sym] = max(short_pos_peak.get(sym, bar.high), bar.high)
                short_pos_trough[sym] = min(short_pos_trough.get(sym, bar.low), bar.low)

            atr_val = atr_indicator.update(bar)
            vwap_val = vwap_indicator.update(bar)

            # Fill pending entry at this bar's open + slippage
            if sym in pending_entries:
                pending = pending_entries.pop(sym)
                size = pending["size"]
                size.shares, entry_slip = self._liquidity_fill(int(size.shares), bar)
                if size.shares <= 0:
                    equity_curve.append((bar.timestamp, portfolio.equity))
                    continue
                fill_price = round(bar.open * (1 + entry_slip), 2)
                atr_entry = pending["atr_val"]
                stop_price = size.long_stop(fill_price)
                target_price = size.long_target(fill_price)
                position = Position(
                    ticker=sym,
                    direction="long",
                    shares=size.shares,
                    entry_price=fill_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    entry_time=bar.timestamp,
                    atr_at_entry=atr_entry,
                    signals=["momentum"],
                    sector="Unknown",
                    highest_close=fill_price,
                    entry_bar_volume=bar.volume,
                )
                portfolio.add_position(position)
                open_records[sym] = TradeRecord(
                    ticker=sym,
                    direction="long",
                    entry_time=bar.timestamp,
                    entry_price=fill_price,
                    shares=size.shares,
                    stop_price=stop_price,
                    target_price=target_price,
                    signals=["momentum"],
                    sector="Unknown",
                    regime="",
                    portfolio_heat_at_entry=portfolio.portfolio_heat_pct,
                    expected_slippage_pct=self._slippage_pct,
                )
                pos_peak[sym] = bar.high
                pos_trough[sym] = bar.low
                logger.debug("BT ENTRY %s @ %.2f shares=%d", sym, fill_price, size.shares)
                equity_curve.append((bar.timestamp, portfolio.equity))
                continue

            # EOD force-close at 15:55 ET
            if bar_et.hour == _EOD_HOUR and bar_et.minute == _EOD_MINUTE:
                if sym in portfolio.positions:
                    should_hold = (
                        self._overnight_holds
                        and vwap_val is not None
                        and baseline > 0
                        and manager.should_hold_overnight(bar, portfolio.positions[sym], vwap_val, baseline)
                    )
                    if not should_hold:
                        self._close(sym, bar.close, "eod", bar.timestamp,
                                    portfolio, open_records, closed_trades,
                                    pos_peak, pos_trough, exit_bar=bar)
                # Shorts are always closed EOD — no overnight holds
                if short_portfolio is not None and sym in short_portfolio.positions:
                    self._close(sym, bar.close, "eod", bar.timestamp,
                                short_portfolio, short_open_records, closed_trades,
                                short_pos_peak, short_pos_trough, exit_bar=bar)
                equity_curve.append((bar.timestamp, portfolio.equity))
                continue

            # Armed pullback limit: fill if price touches the limit, else expire on TTL
            if sym in armed_limits and bar_et.time() < dtime(15, 55):
                armed = armed_limits[sym]
                fill_price = None
                if bar.open <= armed["limit"]:
                    fill_price = bar.open      # gapped below limit: filled at better price
                elif bar.low <= armed["limit"]:
                    fill_price = armed["limit"]
                if fill_price is not None:
                    del armed_limits[sym]
                    kill_switch.check(portfolio, bar.timestamp)
                    can_enter, _ = portfolio.can_enter(sector="Unknown", now=bar.timestamp)
                    if can_enter:
                        atr_armed = armed["atr"]
                        size = compute_position_size(portfolio.equity, atr_armed, fill_price, self._config)
                        if armed["mult"] > 1.0:
                            max_shares = int(portfolio.equity * self._config.max_position_pct / fill_price)
                            size.shares = min(int(size.shares * armed["mult"]), max(0, max_shares))
                        if self._max_position_dv_pct > 0 and baseline > 0:
                            dv_cap = int(self._max_position_dv_pct * baseline * 390)
                            size.shares = min(int(size.shares), dv_cap)
                        # Limit order: no impact slippage past the limit price, but the
                        # bar's volume still bounds how many shares can realistically fill.
                        size.shares, _ = self._liquidity_fill(int(size.shares), bar)
                        if size.shares > 0:
                            stop_price = size.long_stop(fill_price)
                            target_price = size.long_target(fill_price)
                            if self._pullback_target_atr > 0:
                                target_price = round(fill_price + self._pullback_target_atr * atr_armed, 2)
                            position = Position(
                                ticker=sym, direction="long", shares=size.shares,
                                entry_price=fill_price, stop_price=stop_price,
                                target_price=target_price, entry_time=bar.timestamp,
                                atr_at_entry=atr_armed, signals=["momentum_pullback"],
                                sector="Unknown", highest_close=fill_price,
                                entry_bar_volume=bar.volume,
                            )
                            portfolio.add_position(position)
                            open_records[sym] = TradeRecord(
                                ticker=sym, direction="long", entry_time=bar.timestamp,
                                entry_price=fill_price, shares=size.shares,
                                stop_price=stop_price, target_price=target_price,
                                signals=["momentum_pullback"], sector="Unknown", regime="",
                                portfolio_heat_at_entry=portfolio.portfolio_heat_pct,
                                expected_slippage_pct=0.0,
                            )
                            pos_peak[sym] = bar.high
                            pos_trough[sym] = bar.low
                            logger.debug("BT ENTRY (pullback) %s @ %.2f shares=%d",
                                         sym, fill_price, size.shares)
                    equity_curve.append((bar.timestamp, portfolio.equity))
                    continue
                armed["ttl"] -= 1
                if armed["ttl"] <= 0:
                    del armed_limits[sym]   # expired unfilled

            # Exit logic for open long positions (order mirrors live runner _on_bar)
            if sym in portfolio.positions:
                position = portfolio.positions[sym]
                if bar.low <= position.stop_price:
                    # Gap-through: if bar opened below stop, fill at open (realistic)
                    stop_fill = min(position.stop_price, bar.open)
                    self._close(sym, stop_fill, "hard_stop", bar.timestamp,
                                portfolio, open_records, closed_trades,
                                pos_peak, pos_trough, exit_bar=bar)
                elif bar.high >= position.target_price:
                    self._close(sym, position.target_price, "target", bar.timestamp,
                                portfolio, open_records, closed_trades,
                                pos_peak, pos_trough, exit_bar=bar)
                elif vwap_val is not None and baseline > 0:
                    instruction = manager.on_bar(bar, position, vwap_val, baseline)
                    if instruction:
                        if instruction.reason in ("hard_stop", "trailing_stop"):
                            fill = min(position.stop_price, bar.close)
                        elif instruction.limit_price is not None:
                            fill = instruction.limit_price
                        else:
                            fill = bar.close
                        self._close(sym, fill, instruction.reason, bar.timestamp,
                                    portfolio, open_records, closed_trades,
                                    pos_peak, pos_trough, exit_bar=bar)
                equity_curve.append((bar.timestamp, portfolio.equity))
                continue

            # Exit logic for open short positions
            # Stop is ABOVE entry (price rises to stop); target is BELOW entry.
            # No soft exits / trailing for now — hard stop + target only.
            if short_portfolio is not None and sym in short_portfolio.positions:
                position = short_portfolio.positions[sym]
                if bar.high >= position.stop_price:
                    stop_fill = max(position.stop_price, bar.open)
                    self._close(sym, stop_fill, "hard_stop", bar.timestamp,
                                short_portfolio, short_open_records, closed_trades,
                                short_pos_peak, short_pos_trough, exit_bar=bar)
                elif bar.low <= position.target_price:
                    self._close(sym, position.target_price, "target", bar.timestamp,
                                short_portfolio, short_open_records, closed_trades,
                                short_pos_peak, short_pos_trough, exit_bar=bar)
                equity_curve.append((bar.timestamp, portfolio.equity))
                continue

            # Entry logic
            if sym in pending_entries:
                continue
            if sym in traded_today:
                continue

            # Gap-hold entry: fires on first bar after hold period confirmed
            if self._gap_hold_entry and sym in gap_confirmed and sym not in portfolio.positions:
                if atr_val and baseline > 0:
                    _gh_news_ok = not _news_active or catalyst_status.get(sym, False)
                    if not _gh_news_ok:
                        gap_confirmed.discard(sym)
                    else:
                        kill_switch.check(portfolio, bar.timestamp)
                        can_enter, _ = portfolio.can_enter(sector="Unknown", now=bar.timestamp)
                        if can_enter:
                            size = compute_position_size(portfolio.equity, atr_val, bar.close, self._config)
                            if self._max_position_dv_pct > 0 and baseline > 0:
                                dv_cap = int(self._max_position_dv_pct * baseline * 390)
                                size.shares = min(int(size.shares), dv_cap)
                            size.shares, entry_slip = self._liquidity_fill(int(size.shares), bar)
                            if size.shares > 0:
                                fill_price = round(bar.close * (1 + entry_slip), 2)
                                stop_price = size.long_stop(fill_price)
                                target_price = size.long_target(fill_price)
                                position = Position(
                                    ticker=sym, direction="long", shares=size.shares,
                                    entry_price=fill_price, stop_price=stop_price,
                                    target_price=target_price, entry_time=bar.timestamp,
                                    atr_at_entry=atr_val, signals=["gap_hold"],
                                    sector="Unknown", highest_close=fill_price,
                                    entry_bar_volume=bar.volume,
                                )
                                portfolio.add_position(position)
                                open_records[sym] = TradeRecord(
                                    ticker=sym, direction="long", entry_time=bar.timestamp,
                                    entry_price=fill_price, shares=size.shares,
                                    stop_price=stop_price, target_price=target_price,
                                    signals=["gap_hold"], sector="Unknown", regime="",
                                    portfolio_heat_at_entry=portfolio.portfolio_heat_pct,
                                    expected_slippage_pct=self._slippage_pct,
                                )
                                pos_peak[sym] = bar.high
                                pos_trough[sym] = bar.low
                                traded_today.add(sym)
                                gap_confirmed.discard(sym)
                                logger.debug("BT ENTRY (gap-hold) %s @ %.2f shares=%d",
                                             sym, fill_price, size.shares)
                                equity_curve.append((bar.timestamp, portfolio.equity))
                                continue

            if not (atr_val and baseline > 0):
                continue
            kill_switch.check(portfolio, bar.timestamp)  # mutates state if triggered
            can_enter_long, _ = portfolio.can_enter(sector="Unknown", now=bar.timestamp)
            if can_enter_long and validator.validate(bar, baseline):
                if self._require_above_vwap_at_entry and (vwap_val is None or bar.close <= vwap_val):
                    continue
                if self._max_entry_bar_range_pct > 0 and bar.close > 0:
                    if (bar.high - bar.low) / bar.close > self._max_entry_bar_range_pct:
                        continue
                if self._stage2_min_dist_from_day_high_pct > 0:
                    day_high = day_highs.get(sym, bar.high)
                    if day_high > 0 and (day_high - bar.close) / day_high < self._stage2_min_dist_from_day_high_pct:
                        continue
                if self._max_range_position_pct > 0:
                    d_high = day_highs.get(sym, bar.high)
                    d_low = day_lows.get(sym, bar.low)
                    d_range = d_high - d_low
                    if d_range > 0 and (bar.close - d_low) / d_range > self._max_range_position_pct:
                        continue
                # Volume acceleration: require entry bar volume >= X% of previous bar volume.
                # Avoids entering after the momentum spike has already peaked.
                _vol_accel = getattr(self._config, "stage2_min_vol_vs_prev_bar", 0.0)
                if _vol_accel > 0:
                    _prev = prev_bar.get(sym)
                    if _prev is not None and _prev.volume > 0:
                        if bar.volume < _prev.volume * _vol_accel:
                            continue
                # Minimum tier gate (used for regime-bypass mode)
                if self._min_entry_tier > 0:
                    conf = validator.confidence_score(bar, baseline)
                    if self._config.confidence_multiplier(conf) < self._min_entry_tier:
                        continue
                # News gate with optional bypass
                if _news_active and not catalyst_status.get(sym, False):
                    rel_vol = bar.volume / baseline if baseline > 0 else 0.0
                    if self._news_relvol_bypass > 0 and rel_vol >= self._news_relvol_bypass:
                        pass  # rel-vol bypass approved
                    elif self._news_tier_bypass > 0:
                        conf = validator.confidence_score(bar, baseline)
                        mult_check = self._config.confidence_multiplier(conf)
                        if mult_check < self._news_tier_bypass:
                            continue  # insufficient tier — blocked
                    else:
                        continue  # no bypass — blocked
                # Pullback mode: arm a limit below the signal close instead of entering
                # here — unless the signal is at/above pullback_chase_tier, in which
                # case fall through to the immediate market entry below (hybrid mode).
                # tier4_only: skip sub-threshold signals entirely instead of arming.
                if self._pullback_entry_atr > 0:
                    _mult = self._config.confidence_multiplier(
                        validator.confidence_score(bar, baseline)
                    )
                    _below_chase = not (self._pullback_chase_tier > 0 and _mult >= self._pullback_chase_tier)
                    if _below_chase and self._tier4_only:
                        continue  # tier4_only: discard sub-chase signals entirely
                    if _below_chase:
                        armed_limits[sym] = {
                            "limit": round(bar.close - self._pullback_entry_atr * atr_val, 2),
                            "atr": atr_val,
                            "mult": _mult,
                            "ttl": self._pullback_entry_ttl_bars,
                        }
                        traded_today.add(sym)
                        logger.debug("BT ARM (pullback) %s limit=%.2f", sym, armed_limits[sym]["limit"])
                        continue
                size = compute_position_size(portfolio.equity, atr_val, bar.close, self._config)
                if size.shares > 0:
                    mult = self._config.confidence_multiplier(
                        validator.confidence_score(bar, baseline)
                    )
                    if mult > 1.0:
                        max_shares = int(portfolio.equity * self._config.max_position_pct / bar.close)
                        size.shares = min(int(size.shares * mult), max(0, max_shares))
                    # Cap position to a fraction of avg daily dollar volume (liquidity guard)
                    if self._max_position_dv_pct > 0 and baseline > 0:
                        dv_cap = int(self._max_position_dv_pct * baseline * 390)
                        size.shares = min(int(size.shares), dv_cap)
                    if self._market_order_fill:
                        # Market order: fill at signal bar's close + slippage.
                        size.shares, entry_slip = self._liquidity_fill(int(size.shares), bar)
                        if size.shares <= 0:
                            continue
                        fill_price = round(bar.close * (1 + entry_slip), 2)
                        stop_price = size.long_stop(fill_price)
                        target_price = size.long_target(fill_price)
                        position = Position(
                            ticker=sym, direction="long", shares=size.shares,
                            entry_price=fill_price, stop_price=stop_price,
                            target_price=target_price, entry_time=bar.timestamp,
                            atr_at_entry=atr_val, signals=["momentum"],
                            sector="Unknown", highest_close=fill_price,
                            entry_bar_volume=bar.volume,
                        )
                        portfolio.add_position(position)
                        open_records[sym] = TradeRecord(
                            ticker=sym, direction="long", entry_time=bar.timestamp,
                            entry_price=fill_price, shares=size.shares,
                            stop_price=stop_price, target_price=target_price,
                            signals=["momentum"], sector="Unknown", regime="",
                            portfolio_heat_at_entry=portfolio.portfolio_heat_pct,
                            expected_slippage_pct=self._slippage_pct,
                        )
                        pos_peak[sym] = bar.high
                        pos_trough[sym] = bar.low
                        traded_today.add(sym)
                        logger.debug("BT ENTRY (market) %s @ %.2f shares=%d", sym, fill_price, size.shares)
                        equity_curve.append((bar.timestamp, portfolio.equity))
                    else:
                        pending_entries[sym] = {"atr_val": atr_val, "size": size}
                        traded_today.add(sym)
                        logger.debug("BT SIGNAL %s @ %s", sym, bar.timestamp)

            # --- Short entry ---
            # Same bar (same long signal bar): after long entry check, try a short.
            # We use the same validator signal (the stock is a mover) but look for
            # exhaustion instead of continuation.  Block if:
            #   - short_config not set
            #   - already long or short this symbol today
            #   - ETB set provided and symbol not in it
            #   - SSR: close has fallen ≥10% from prior close (uptick rule in effect)
            if (
                short_portfolio is not None
                and sym not in short_traded_today
                and sym not in traded_today              # no short if long entry already today
                and sym not in pending_entries           # no short if long entry pending
                and sym not in portfolio.positions       # no short if long position open
                and sym not in short_portfolio.positions
                and atr_val
                and baseline > 0
            ):
                scfg = self._short_config
                assert scfg is not None  # narrowing for type checkers

                # ETB gate: skip if we have a populated ETB set and sym isn't in it
                if self._etb_set and sym not in self._etb_set:
                    pass
                else:
                    # SSR block: if price is down ≥10% from prior close, skip
                    _prev_close = (prev_closes or {}).get(sym, 0.0)
                    _ssr_active = (
                        _prev_close > 0
                        and bar.close <= _prev_close * 0.90
                    )
                    if not _ssr_active and bar.close >= scfg.stage1_min_price:
                        if baseline * 390 * bar.close >= scfg.min_avg_dollar_volume:
                            if self._validate_short_entry(bar, prev_bar.get(sym), scfg):
                                can_enter_short, _ = short_portfolio.can_enter(
                                    sector="Unknown", now=bar.timestamp
                                )
                                if can_enter_short:
                                    size = compute_position_size(
                                        short_portfolio.equity, atr_val, bar.close, scfg
                                    )
                                    size.shares, entry_slip = self._liquidity_fill(int(size.shares), bar)
                                    if size.shares > 0:
                                        # Fill next bar open (like longs) — sell short with slippage
                                        fill_price = round(bar.close * (1 - entry_slip), 2)
                                        stop_price = size.short_stop(fill_price)
                                        target_price = size.short_target(fill_price)
                                        position = Position(
                                            ticker=sym, direction="short",
                                            shares=size.shares,
                                            entry_price=fill_price,
                                            stop_price=stop_price,
                                            target_price=target_price,
                                            entry_time=bar.timestamp,
                                            atr_at_entry=atr_val,
                                            signals=["short_fade"],
                                            sector="Unknown",
                                            highest_close=fill_price,
                                            entry_bar_volume=bar.volume,
                                        )
                                        short_portfolio.add_position(position)
                                        short_open_records[sym] = TradeRecord(
                                            ticker=sym, direction="short",
                                            entry_time=bar.timestamp,
                                            entry_price=fill_price,
                                            shares=size.shares,
                                            stop_price=stop_price,
                                            target_price=target_price,
                                            signals=["short_fade"],
                                            sector="Unknown", regime="",
                                            portfolio_heat_at_entry=short_portfolio.portfolio_heat_pct,
                                            expected_slippage_pct=self._slippage_pct,
                                        )
                                        short_pos_peak[sym] = bar.high
                                        short_pos_trough[sym] = bar.low
                                        short_traded_today.add(sym)
                                        logger.debug(
                                            "BT SHORT ENTRY %s @ %.2f shares=%d",
                                            sym, fill_price, size.shares,
                                        )

        # Close any positions that never received a 15:55 bar (halted/illiquid stocks).
        # In live the EOD timer thread handles this; without it the sim would silently
        # drop these trades, biasing results by omitting end-of-day disaster scenarios.
        for sym in list(portfolio.positions.keys()):
            lb = last_bar.get(sym)
            if lb is None:
                continue
            self._close(sym, lb.close, "eod_no_bar", lb.timestamp,
                        portfolio, open_records, closed_trades,
                        pos_peak, pos_trough)
            equity_curve.append((lb.timestamp, portfolio.equity))

        if short_portfolio is not None:
            for sym in list(short_portfolio.positions.keys()):
                lb = last_bar.get(sym)
                if lb is None:
                    continue
                self._close(sym, lb.close, "eod_no_bar", lb.timestamp,
                            short_portfolio, short_open_records, closed_trades,
                            short_pos_peak, short_pos_trough)

        return BacktestResult(trades=closed_trades, equity_curve=equity_curve)

    @staticmethod
    def _validate_short_entry(bar: Bar, prev: Optional[Bar], cfg: V4Config) -> bool:
        """Return True if bar meets the short exhaustion entry criteria."""
        bar_range = bar.high - bar.low
        # Selling pressure: close must be in bottom fraction of bar's range
        if cfg.short_selling_pressure_max < 1.0 and bar_range > 0:
            pressure = (bar.close - bar.low) / bar_range
            if pressure >= cfg.short_selling_pressure_max:
                return False
        # Red bar: close must be below open
        if cfg.short_require_red_bar and bar.close >= bar.open:
            return False
        # Volume exhaustion: volume must be declining vs prior bar
        if cfg.short_volume_exhaustion_ratio > 0 and prev is not None and prev.volume > 0:
            if bar.volume >= prev.volume * cfg.short_volume_exhaustion_ratio:
                return False
        return True

    def _close(
        self,
        sym: str,
        fill_price: float,
        reason: str,
        ts: datetime,
        portfolio: PortfolioState,
        open_records: Dict[str, TradeRecord],
        closed_trades: List[TradeRecord],
        pos_peak: Optional[Dict[str, float]] = None,
        pos_trough: Optional[Dict[str, float]] = None,
        exit_bar: Optional[Bar] = None,
    ) -> None:
        position = portfolio.remove_position(sym)
        if not position:
            return
        exit_slip = self._exit_slippage(position, exit_bar)
        if exit_slip > 0:
            if position.direction == "short":
                fill_price = round(fill_price * (1 + exit_slip), 4)
            else:
                fill_price = round(fill_price * (1 - exit_slip), 4)
        if position.direction == "short":
            pnl = round((position.entry_price - fill_price) * position.shares, 2)
        else:
            pnl = round((fill_price - position.entry_price) * position.shares, 2)
        portfolio.equity += pnl
        if pnl < 0:
            portfolio.consecutive_losses += 1
        else:
            portfolio.consecutive_losses = 0
        record = open_records.pop(sym, None)
        if record:
            record.exit_time = ts
            record.exit_price = fill_price
            record.pnl = pnl
            record.exit_reason = reason
            ep = record.entry_price
            if ep > 0:
                if position.direction == "short":
                    # For shorts: MFE = how far price fell, MAE = how far price rose
                    if pos_trough and sym in pos_trough:
                        record.mfe_pct = (ep - pos_trough[sym]) / ep * 100
                    if pos_peak and sym in pos_peak:
                        record.mae_pct = (pos_peak[sym] - ep) / ep * 100
                else:
                    if pos_peak and sym in pos_peak:
                        record.mfe_pct = (pos_peak[sym] - ep) / ep * 100
                    if pos_trough and sym in pos_trough:
                        record.mae_pct = (pos_trough[sym] - ep) / ep * 100
            closed_trades.append(record)
        logger.debug("BT EXIT %s @ %.2f pnl=%.2f reason=%s", sym, fill_price, pnl, reason)
