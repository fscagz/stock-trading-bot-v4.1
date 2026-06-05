from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from bot.backtest.news_filter import NewsFilter
from bot.config import V4Config
from bot.intraday.indicators.atr import ATRIndicator
from bot.intraday.indicators.vwap import VWAPIndicator
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
        self._max_position_dv_pct = max_position_dv_pct

    def run_day(
        self,
        trade_date: date,
        bars_by_symbol: Dict[str, List[Bar]],
        baseline_volumes: Dict[str, float],
    ) -> BacktestResult:
        atr_indicator = ATRIndicator(period=14)
        vwap_indicator = VWAPIndicator()
        validator = MomentumValidator(self._config)
        manager = PositionManager(self._config)
        portfolio = PortfolioState(equity=self._initial_equity, config=self._config)

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
        traded_today: set = set()  # symbols that already had an entry today
        equity_curve: List[Tuple[datetime, float]] = []

        for bar in merged:
            sym = bar.symbol
            baseline = baseline_volumes.get(sym, 0.0)
            bar_et = bar.timestamp.astimezone(_ET)

            atr_val = atr_indicator.update(bar)
            vwap_val = vwap_indicator.update(bar)

            # Fill pending entry at this bar's open + slippage
            if sym in pending_entries:
                pending = pending_entries.pop(sym)
                fill_price = round(bar.open * (1 + self._slippage_pct), 2)
                size = pending["size"]
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
                logger.debug("BT ENTRY %s @ %.2f shares=%d", sym, fill_price, size.shares)
                equity_curve.append((bar.timestamp, portfolio.equity))
                continue

            # EOD evaluation at 15:25 ET
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
                                    portfolio, open_records, closed_trades)
                equity_curve.append((bar.timestamp, portfolio.equity))
                continue

            # Exit logic for open positions
            if sym in portfolio.positions:
                position = portfolio.positions[sym]
                if bar.low <= position.stop_price:
                    # Mid-bar stop hit: fill at stop_price
                    self._close(sym, position.stop_price, "hard_stop", bar.timestamp,
                                portfolio, open_records, closed_trades)
                elif vwap_val is not None and baseline > 0:
                    instruction = manager.on_bar(bar, position, vwap_val, baseline)
                    if instruction:
                        if instruction.reason in ("hard_stop", "trailing_stop"):
                            fill = position.stop_price
                        elif instruction.limit_price is not None:
                            fill = instruction.limit_price
                        else:
                            fill = bar.close
                        self._close(sym, fill, instruction.reason, bar.timestamp,
                                    portfolio, open_records, closed_trades)
                equity_curve.append((bar.timestamp, portfolio.equity))
                continue

            # Entry logic
            if sym in pending_entries:
                continue
            if sym in traded_today:
                continue
            if not (atr_val and baseline > 0):
                continue
            can_enter, _ = portfolio.can_enter(sector="Unknown", now=bar.timestamp)
            if not can_enter:
                continue
            if validator.validate(bar, baseline):
                if self._require_above_vwap_at_entry and (vwap_val is None or bar.close <= vwap_val):
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
                        # Market order: fill immediately at signal bar's close, no slippage.
                        # Matches live runner behaviour (market order submitted at bar close).
                        fill_price = bar.close
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
                            expected_slippage_pct=0.0,
                        )
                        traded_today.add(sym)
                        logger.debug("BT ENTRY (market) %s @ %.2f shares=%d", sym, fill_price, size.shares)
                        equity_curve.append((bar.timestamp, portfolio.equity))
                    else:
                        pending_entries[sym] = {"atr_val": atr_val, "size": size}
                        traded_today.add(sym)
                        logger.debug("BT SIGNAL %s @ %s", sym, bar.timestamp)

        return BacktestResult(trades=closed_trades, equity_curve=equity_curve)

    def _close(
        self,
        sym: str,
        fill_price: float,
        reason: str,
        ts: datetime,
        portfolio: PortfolioState,
        open_records: Dict[str, TradeRecord],
        closed_trades: List[TradeRecord],
    ) -> None:
        position = portfolio.remove_position(sym)
        if not position:
            return
        pnl = round((fill_price - position.entry_price) * position.shares, 2)
        portfolio.equity += pnl
        record = open_records.pop(sym, None)
        if record:
            record.exit_time = ts
            record.exit_price = fill_price
            record.pnl = pnl
            record.exit_reason = reason
            closed_trades.append(record)
        logger.debug("BT EXIT %s @ %.2f pnl=%.2f reason=%s", sym, fill_price, pnl, reason)
