"""
Central configuration for the systematic portfolio pipeline.

Universe, time horizons, paths, and run mode. Override via environment
variables or by editing this file. See future plan.md for rationale.
"""

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("BOT_DATA_DIR", str(PROJECT_ROOT / "data")))
CACHE_DIR = Path(os.getenv("BOT_CACHE_DIR", str(DATA_DIR / "cache")))
UNIVERSE_SNAPSHOT_DIR = DATA_DIR / "universe_snapshots"

# -----------------------------------------------------------------------------
# Investment universe (plan: US large/mid cap, min liquidity, no microcaps)
# -----------------------------------------------------------------------------
UNIVERSE_SOURCE = os.getenv("BOT_UNIVERSE_SOURCE", "sp500")  # sp500, custom, etc.
UNIVERSE_REFRESH_QUARTERLY = True
MIN_AVG_DAILY_VOLUME = int(os.getenv("BOT_MIN_ADV", "500_000").replace("_", ""))  # shares
MIN_MARKET_CAP = None  # optional: float (e.g. 1e9); set when data available

# -----------------------------------------------------------------------------
# Time horizon (plan: daily data, 1–3 month forward return, monthly rebalance)
# -----------------------------------------------------------------------------
SIGNAL_FREQUENCY = "daily"
FORWARD_RETURN_MONTHS = (1, 2, 3)  # prediction horizons
REBALANCE_FREQ = os.getenv("BOT_REBALANCE_FREQ", "monthly")  # monthly | quarterly
REBALANCE_DAY = int(os.getenv("BOT_REBALANCE_DAY", "1"))  # day of month (1 = first trading day)

# -----------------------------------------------------------------------------
# Model (factor-only baseline first; ML added later)
# -----------------------------------------------------------------------------
MODEL_MODE = os.getenv("BOT_MODEL_MODE", "factor")  # factor | ml
FACTOR_TOP_PCT = float(os.getenv("BOT_FACTOR_TOP_PCT", "0.20"))  # long top 20%
FACTOR_TOP_N = None  # optional: int; if set, overrides top percentile

# -----------------------------------------------------------------------------
# Risk (placeholders; full logic in Phase 5)
# -----------------------------------------------------------------------------
MAX_POSITION_PCT = float(os.getenv("BOT_MAX_POSITION_PCT", "0.05"))  # 5% cap per name
MAX_SECTOR_PCT = float(os.getenv("BOT_MAX_SECTOR_PCT", "0.40"))  # 40% per sector
TARGET_VOL = None  # optional: annualized vol target
DRAWDOWN_REDUCE_PCT = float(os.getenv("BOT_DRAWDOWN_REDUCE_PCT", "0.10"))  # cut exposure at 10% DD

# -----------------------------------------------------------------------------
# Run mode (plan: research → paper → live)
# -----------------------------------------------------------------------------
RUN_MODE = os.getenv("BOT_RUN_MODE", "backtest")  # backtest | paper | live
PAPER_TRADING = os.getenv("APCA_PAPER", "true").lower() == "true"  # Alpaca paper vs live

# -----------------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------------
BACKTEST_START = os.getenv("BOT_BACKTEST_START", "2010-01-01")
BACKTEST_END = os.getenv("BOT_BACKTEST_END", "")  # empty = latest
TRANSACTION_COST_BPS = float(os.getenv("BOT_TCOST_BPS", "10"))  # basis points per trade

# -----------------------------------------------------------------------------
# Data source flags (swap sources without touching feature code)
# -----------------------------------------------------------------------------
USE_SIMFIN = os.getenv("BOT_USE_SIMFIN", "true").lower() == "true"
USE_EDGAR = os.getenv("BOT_USE_EDGAR", "false").lower() == "true"
USE_EODHD = os.getenv("BOT_USE_EODHD", "false").lower() == "true"
USE_FRED = os.getenv("BOT_USE_FRED", "false").lower() == "true"

SIMFIN_API_KEY = os.getenv("SIMFIN_API_KEY", "")
EODHD_API_KEY = os.getenv("EODHD_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_dirs():
    """Create data and cache directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    UNIVERSE_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def get_rebalance_freq_months():
    """Return number of months between rebalances."""
    return 1 if REBALANCE_FREQ == "monthly" else 3
