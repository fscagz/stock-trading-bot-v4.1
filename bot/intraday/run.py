"""
V3 Paper Trading Runner.

Usage:
    python -m bot.intraday.run_paper

Requires environment variables:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY

Optional environment variables:
    INTRADAY_MODEL_PATH   path to trained MLScorer pickle; enables ML sizing
"""
import logging
import os

from dotenv import load_dotenv

from bot.intraday.bot import IntradayBot
from bot.intraday.config import IntradayConfig
from bot.intraday.data.market_snapshot import compute_adx
from bot.intraday.data.sectors import fetch_sector_map
from bot.intraday.data.correlation import compute_correlation_matrix
from bot.intraday.data.universe_loader import get_candidate_symbols, screen_symbols
from bot.intraday.risk.regime import Regime, RegimeDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def _get_regime(config: IntradayConfig) -> Regime:
    """Classify today's opening regime from SPY and VIX data."""
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY")
        vix = yf.Ticker("^VIX")
        spy_hist = spy.history(period="60d")
        vix_hist = vix.history(period="5d")

        spy_price = float(spy_hist["Close"].iloc[-1])
        spy_ma50 = float(spy_hist["Close"].tail(50).mean())
        vix_level = float(vix_hist["Close"].iloc[-1])
        adx = compute_adx(spy_hist, period=14)

        spy_gap_pct = 0.0
        if len(spy_hist) >= 2:
            prev_close = float(spy_hist["Close"].iloc[-2])
            spy_gap_pct = abs(spy_price - prev_close) / prev_close

        regime = RegimeDetector(config).classify(
            spy_price=spy_price, spy_ma50=spy_ma50,
            vix=vix_level, adx=adx, spy_gap_pct=spy_gap_pct,
        )
        logging.info(
            "Regime: %s (SPY=%.2f MA50=%.2f VIX=%.1f ADX=%.1f)",
            regime.value, spy_price, spy_ma50, vix_level, adx,
        )
        return regime
    except Exception as exc:
        logging.warning("Could not determine regime (%s); defaulting to RANGE_BOUND", exc)
        return Regime.RANGE_BOUND


def main() -> None:
    load_dotenv()

    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        raise RuntimeError("alpaca-py required: pip install alpaca-py")

    api_key = os.environ["APCA_API_KEY_ID"]
    secret_key = os.environ["APCA_API_SECRET_KEY"]
    model_path = os.environ.get("INTRADAY_MODEL_PATH")

    broker = TradingClient(api_key, secret_key)
    account = broker.get_account()
    equity = float(account.equity)
    logging.info("Account equity: $%.2f", equity)

    config = IntradayConfig()
    regime = _get_regime(config)

    logging.info("Building dynamic universe...")
    candidates = get_candidate_symbols()
    symbols = screen_symbols(candidates, config)
    if not symbols:
        raise RuntimeError("Universe screening returned no symbols — check network and config thresholds")
    logging.info("Trading universe: %d symbols", len(symbols))

    logging.info("Fetching sector map...")
    sector_map = fetch_sector_map(symbols)

    logging.info("Computing correlation matrix...")
    correlation_map = compute_correlation_matrix(symbols)

    bot = IntradayBot(
        config=config,
        broker=broker,
        symbols=symbols,
        trade_log_path="bot/trade_log.csv",
        sector_map=sector_map,
        model_path=model_path,
        correlation_map=correlation_map,
    )
    bot.initialize_session(equity=equity, regime=regime)
    bot.start()


if __name__ == "__main__":
    main()
