from __future__ import annotations
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Dict
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

from dotenv import load_dotenv

import bot.broker_alpaca as broker
from bot.config import V4Config
from bot.intraday.data.stream import BarStream
from bot.intraday.indicators.atr import ATRIndicator
from bot.intraday.indicators.vwap import VWAPIndicator
from bot.intraday.risk.kill_switch import KillSwitch
from bot.intraday.risk.portfolio import PortfolioState
from bot.intraday.risk.sizing import compute_position_size
from bot.intraday.types import Bar, Position, TradeRecord
from bot.momentum.validator import MomentumValidator
from bot.positions.manager import ExitInstruction, PositionManager
from bot.scanner.market_scanner import MarketScanner
from bot.scanner.watchlist import Watchlist
from bot.trade_logger import TradeLogger

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def _execute_exit(instruction: ExitInstruction, position: Position) -> None:
    sym = position.ticker
    if instruction.action == "market_exit":
        broker.submit_market_order(sym, position.shares, side="sell")
        logger.info("MARKET EXIT %s — reason=%s", sym, instruction.reason)
    elif instruction.action == "limit_exit" and instruction.limit_price:
        broker.submit_limit_order(sym, position.shares, "sell", instruction.limit_price)
        logger.info("LIMIT EXIT %s @ %.2f — reason=%s", sym, instruction.limit_price, instruction.reason)

    if position.stop_order_id:
        try:
            broker.cancel_order(position.stop_order_id)
        except Exception as exc:
            logger.warning("Could not cancel stop order %s: %s", position.stop_order_id, exc)


def _wait_for_fill(order_id: str, timeout: int) -> tuple[bool, float]:
    for _ in range(timeout):
        try:
            order = broker.get_order(order_id)
            if str(order.status) == "filled":
                return True, float(order.filled_avg_price)
        except Exception:
            pass
        time.sleep(1)
    return False, 0.0


def _update_trailing_stop(position: Position, new_stop: float) -> None:
    if position.stop_order_id:
        try:
            broker.cancel_order(position.stop_order_id)
            new_id = broker.submit_stop_order(position.ticker, position.shares, new_stop)
            position.stop_order_id = new_id
            logger.debug("Trailing stop updated for %s: %.2f", position.ticker, new_stop)
        except Exception as exc:
            logger.warning("Trailing stop update failed for %s: %s", position.ticker, exc)


def _close_record(
    record: TradeRecord,
    exit_time: datetime,
    exit_price: float,
    exit_reason: str,
    trade_logger: TradeLogger,
) -> None:
    record.exit_time = exit_time
    record.exit_price = exit_price
    record.pnl = round((exit_price - record.entry_price) * record.shares, 2)
    record.exit_reason = exit_reason
    trade_logger.log(record)
    logger.info(
        "TRADE CLOSED %s: pnl=%.2f reason=%s",
        record.ticker, record.pnl, exit_reason,
    )


def main() -> None:
    load_dotenv()
    api_key = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]

    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    config = V4Config()
    account = broker.get_account_info()
    equity = account["portfolio_value"]

    portfolio = PortfolioState(equity=equity, config=config)
    kill_switch = KillSwitch(config)
    atr_indicator = ATRIndicator(period=14)
    vwap_indicator = VWAPIndicator()
    validator = MomentumValidator(config)
    manager = PositionManager(config)
    trade_logger = TradeLogger(log_dir="logs")

    stream = BarStream(api_key, secret_key, symbols=[])
    watchlist = Watchlist(stream, config)
    scanner = MarketScanner(api_key, secret_key, config, watchlist, base_url=base_url)

    eod_hour, eod_minute = (int(x) for x in config.eod_evaluation.split(":"))

    # Open trade records keyed by ticker, completed and flushed to CSV on exit
    open_records: Dict[str, TradeRecord] = {}

    def on_bar(bar: Bar) -> None:
        now = datetime.now(timezone.utc)
        now_et = now.astimezone(_ET)
        kill_switch.check(portfolio, now)
        if portfolio.kill_switch_active:
            return

        atr_val = atr_indicator.update(bar)
        vwap_val = vwap_indicator.update(bar)
        baseline = watchlist.get_baseline_volume(bar.symbol)

        # --- Exit logic for open positions ---
        if bar.symbol in portfolio.positions:
            position = portfolio.positions[bar.symbol]

            # Overnight hold evaluation at eod_evaluation time (ET)
            if now_et.hour == eod_hour and now_et.minute == eod_minute:
                if not (vwap_val and baseline and
                        manager.should_hold_overnight(bar, position, vwap_val, baseline)):
                    _execute_exit(ExitInstruction(reason="eod", action="market_exit"), position)
                    portfolio.remove_position(bar.symbol)
                    record = open_records.pop(bar.symbol, None)
                    if record:
                        _close_record(record, now, bar.close, "eod", trade_logger)
                return

            if vwap_val and baseline:
                old_stop = position.stop_price
                instruction = manager.on_bar(bar, position, vwap_val, baseline)
                if instruction:
                    exit_price = instruction.limit_price if instruction.limit_price else bar.close
                    _execute_exit(instruction, position)
                    portfolio.remove_position(bar.symbol)
                    record = open_records.pop(bar.symbol, None)
                    if record:
                        _close_record(record, now, exit_price, instruction.reason, trade_logger)
                elif position.stop_price > old_stop:
                    _update_trailing_stop(position, position.stop_price)
            return

        # --- Entry logic for watchlist candidates ---
        if bar.symbol not in watchlist.symbols:
            return
        if not (atr_val and baseline):
            return
        if not validator.validate(bar, baseline):
            return

        can_enter, reason = portfolio.can_enter(sector="Unknown", now=now)
        if not can_enter:
            logger.debug("Entry blocked for %s: %s", bar.symbol, reason)
            return

        size = compute_position_size(portfolio.equity, atr_val, bar.close, config)
        if size.shares <= 0:
            return

        limit_price = round(bar.close * (1 + config.limit_offset_pct), 2)
        try:
            order_id = broker.submit_limit_order(bar.symbol, size.shares, "buy", limit_price)
            filled, fill_price = _wait_for_fill(order_id, config.fill_timeout_seconds)
            if not filled:
                broker.cancel_order(order_id)
                logger.info("Entry unfilled for %s — cancelled", bar.symbol)
                return

            stop_price = size.long_stop(fill_price)
            target_price = size.long_target(fill_price)
            stop_order_id = broker.submit_stop_order(bar.symbol, size.shares, stop_price)

            position = Position(
                ticker=bar.symbol,
                direction="long",
                shares=size.shares,
                entry_price=fill_price,
                stop_price=stop_price,
                target_price=target_price,
                entry_time=now,
                atr_at_entry=atr_val,
                signals=["momentum"],
                sector="Unknown",
                highest_close=fill_price,
                stop_order_id=stop_order_id,
                entry_bar_volume=bar.volume,
            )
            portfolio.add_position(position)

            open_records[bar.symbol] = TradeRecord(
                ticker=bar.symbol,
                direction="long",
                entry_time=now,
                entry_price=fill_price,
                shares=size.shares,
                stop_price=stop_price,
                target_price=target_price,
                signals=["momentum"],
                sector="Unknown",
                regime="",
                portfolio_heat_at_entry=portfolio.portfolio_heat_pct,
                expected_slippage_pct=config.expected_entry_slippage_pct,
            )

            logger.info("ENTRY %s: %d shares @ %.2f, stop=%.2f, target=%.2f",
                        bar.symbol, size.shares, fill_price, stop_price, target_price)
        except Exception as exc:
            logger.error("Entry failed for %s: %s", bar.symbol, exc)

    stream.set_handler(on_bar)

    scanner_thread = threading.Thread(target=scanner.run, daemon=True)
    scanner_thread.start()
    logger.info("V4 Momentum Bot started. Equity: $%.2f", equity)

    stream.run()


if __name__ == "__main__":
    main()
