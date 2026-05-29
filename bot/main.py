from __future__ import annotations
import logging
import os
import threading
import time
from datetime import datetime, timezone
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
from bot.intraday.types import Bar, Position
from bot.momentum.validator import MomentumValidator
from bot.positions.manager import ExitInstruction, PositionManager
from bot.scanner.market_scanner import MarketScanner
from bot.scanner.watchlist import Watchlist

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


def main() -> None:
    load_dotenv()
    api_key = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]

    config = V4Config()
    account = broker.get_account_info()
    equity = account["portfolio_value"]

    portfolio = PortfolioState(equity=equity, config=config)
    kill_switch = KillSwitch(config)
    atr_indicator = ATRIndicator(period=14)
    vwap_indicator = VWAPIndicator()
    validator = MomentumValidator(config)
    manager = PositionManager(config)

    stream = BarStream(api_key, secret_key, symbols=[])
    watchlist = Watchlist(stream, config)
    scanner = MarketScanner(api_key, secret_key, config, watchlist)

    eod_hour, eod_minute = (int(x) for x in config.eod_evaluation.split(":"))

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
                return

            if vwap_val and baseline:
                old_stop = position.stop_price
                instruction = manager.on_bar(bar, position, vwap_val, baseline)
                if instruction:
                    _execute_exit(instruction, position)
                    portfolio.remove_position(bar.symbol)
                elif position.stop_price > old_stop:
                    # Trailing stop moved up — update broker order
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
