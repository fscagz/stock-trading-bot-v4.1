"""
Pytest path setup.

This repo contains two subsystems with different import conventions:

  * the live micro-cap bot (bot/main.py, bot/config.py, bot/intraday/*) uses
    `bot.`-prefixed imports and expects the REPO ROOT on sys.path;
  * the systematic factor pipeline (bot/backtest/engine.py, bot/data/*,
    bot/risk/*) uses bare imports such as `from data.universe import ...`
    and expects BOT/ on sys.path.

Neither is wrong, but importing the systematic stack fails unless both are
present. Adding both here keeps pytest working for either subsystem without
rewriting hundreds of import statements.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "bot"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
