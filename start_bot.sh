#!/usr/bin/env bash
# Launch the trading bot with output captured to logs/bot.log
# Usage: ./start_bot.sh [--risk-scale 0.5]

cd "$(dirname "$0")"
mkdir -p logs

# Kill any existing bot process
pkill -f "bot.main" 2>/dev/null
sleep 1

echo "[$(date)] Starting bot..." | tee -a logs/bot.log
.venv/bin/python3.11 -m bot.main "$@" 2>&1 | tee -a logs/bot.log
