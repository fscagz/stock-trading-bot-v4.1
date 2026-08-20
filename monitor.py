#!/usr/bin/env python3
"""
Claude AI monitor for the trading bot.

Runs alongside the bot subprocess and:
  1. Bug-fix mode (reactive): tails stdout for Python tracebacks → Claude reads
     the traceback + relevant source file → writes a fix → restarts the bot.
  2. Tuning mode (periodic, every TUNING_INTERVAL_MIN minutes): reads today's
     trade CSV + current config → Claude suggests and applies parameter changes.

Usage:
    python monitor.py [--bot-cmd "python -m bot.main"] [--tune-interval 30]

Set ANTHROPIC_API_KEY in your environment (or .env) before running.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import date
from pathlib import Path

import anthropic
from anthropic import beta_tool
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

_PROJECT_ROOT = Path(__file__).resolve().parent
_LOGS_DIR = _PROJECT_ROOT / "logs"
_BOT_MODULE = "bot.main"

_TRACEBACK_START = "Traceback (most recent call last):"
_LOG_BUFFER_SIZE = 2000  # lines kept in memory
_MIN_LINES_BEFORE_BUG_FIX = 5  # must have at least this many log lines before invoking Claude
_COOLDOWN_AFTER_FIX_SEC = 60  # seconds to wait after a fix before watching for new errors

MODEL = "claude-opus-4-8"
THINKING = {"type": "adaptive"}
MAX_TOKENS = 16384

_log_buffer: deque[str] = deque(maxlen=_LOG_BUFFER_SIZE)
_bot_proc: subprocess.Popen | None = None
_fix_lock = threading.Lock()  # prevent concurrent bug-fix invocations
_last_fix_time: float = 0.0
_bot_cmd: list[str] = []


# ---------------------------------------------------------------------------
# Tool definitions (shared between bug-fix and tuning sessions)
# ---------------------------------------------------------------------------

@beta_tool
def read_file(path: str) -> str:
    """Read any file in the trading-bot project. Path must be relative to the project root."""
    full = _PROJECT_ROOT / path
    if not full.exists():
        return f"ERROR: File not found: {path}"
    try:
        return full.read_text()
    except Exception as exc:
        return f"ERROR reading {path}: {exc}"


@beta_tool
def write_file(path: str, content: str) -> str:
    """Overwrite a file in the trading-bot project with new content. Path relative to project root."""
    full = _PROJECT_ROOT / path
    try:
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        return f"OK: wrote {len(content)} bytes to {path}"
    except Exception as exc:
        return f"ERROR writing {path}: {exc}"


@beta_tool
def read_recent_logs(n_lines: int = 100) -> str:
    """Return the most recent N lines from the bot's captured stdout."""
    lines = list(_log_buffer)[-n_lines:]
    return "\n".join(lines) if lines else "(no log lines captured yet)"


@beta_tool
def read_recent_trades(n: int = 50) -> str:
    """Return the most recent N closed trades from today's trade CSV."""
    today = date.today().strftime("%Y-%m-%d")
    csv_path = _LOGS_DIR / f"trades_{today}.csv"
    if not csv_path.exists():
        return f"No trade log for today ({today}) yet."
    try:
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        recent = rows[-n:]
        if not recent:
            return "Trade log exists but contains no rows."
        lines = [",".join(recent[0].keys())]
        lines += [",".join(str(v) for v in r.values()) for r in recent]
        return "\n".join(lines)
    except Exception as exc:
        return f"ERROR reading trade CSV: {exc}"


@beta_tool
def restart_bot() -> str:
    """Kill the current bot process and restart it. Use after applying a fix."""
    global _bot_proc
    if _bot_proc is not None:
        try:
            _bot_proc.terminate()
            _bot_proc.wait(timeout=10)
        except Exception:
            try:
                _bot_proc.kill()
            except Exception:
                pass
    _bot_proc = _launch_bot()
    return "Bot restarted." if _bot_proc else "ERROR: failed to launch bot."


@beta_tool
def list_files(directory: str = ".") -> str:
    """List files in a directory (relative to project root, max 2 levels deep)."""
    full = _PROJECT_ROOT / directory
    if not full.is_dir():
        return f"ERROR: {directory} is not a directory"
    result = []
    try:
        for p in sorted(full.rglob("*.py")):
            rel = p.relative_to(_PROJECT_ROOT)
            parts = rel.parts
            if len(parts) <= 3:
                result.append(str(rel))
        return "\n".join(result[:100]) if result else "(no .py files found)"
    except Exception as exc:
        return f"ERROR: {exc}"


_ALL_TOOLS = [read_file, write_file, read_recent_logs, read_recent_trades, restart_bot, list_files]

_BUG_FIX_SYSTEM = """\
You are an AI assistant monitoring a live paper-trading bot written in Python.
The bot runs as `python -m bot.main` inside the project root.
You have tools to read source files, write fixed files, read logs, and restart the bot.

When called, you will be given a Python traceback from the bot's stdout.
Your job:
1. Read the relevant source file(s) to understand the bug.
2. Write a minimal, correct fix directly to the source file.
3. Restart the bot by calling restart_bot().
4. Briefly explain what you changed and why.

IMPORTANT:
- Only change what is needed to fix the bug. Do not refactor or clean up unrelated code.
- Preserve all existing logic outside the bug site.
- The bot uses Python 3.13 inside a .venv. All paths passed to read_file/write_file
  are relative to the project root.
- After restarting, your response is done. Do not wait or poll further.
"""

_TUNING_SYSTEM = """\
You are an AI assistant monitoring a live paper-trading momentum bot.
The bot trades intraday on a paper account. You have access to today's trade log
and all source/config files.

Configuration lives in two places:
  - bot/config.py  — V4Config dataclass (extends IntradayConfig from bot/intraday/config.py)
  - bot/main.py    — where the config object is instantiated and individual fields are overridden

When called:
1. Read today's trade CSV to understand recent performance.
2. Read bot/main.py and bot/config.py to understand current settings.
3. Analyze performance: win rate, avg pnl, exit reasons, slippage, heat levels.
4. If you see a clear improvement (e.g., high stop-out rate → widen stop, large slippage → tighten
   entry filters, no trades → loosen screener), apply it by editing bot/main.py and restarting.
5. If unsure, write a plain-text recommendation to logs/tuning_notes.txt instead.
6. Always append your analysis to logs/tuning_notes.txt regardless of whether you make changes.

Tunable parameters in bot/main.py (set after `long_config = make_gap_hold_config()`):
  long_config.risk_per_trade            (default 0.005)
  long_config.stop_atr_multiple         (default 1.5)
  long_config.target_atr_multiple       (default 4.0 for tier-4 chase)
  long_config.stage2_roc_min_pct        (default ~0.03)
  long_config.stage2_min_relative_volume (default ~4.0)
  long_config.stage2_buying_pressure_min (default ~0.75 — 0.85 for gap-hold)
  long_config.stage2_min_dist_from_day_high_pct (default 0.05)
  long_config.stage2_min_vol_vs_prev_bar        (default 0.80)
  long_config.max_portfolio_heat               (default 0.03)
  long_config.max_daily_drawdown               (default 0.02)

Conservative constraints:
- Never set risk_per_trade above 0.01 or max_portfolio_heat above 0.06.
- Never set stop_atr_multiple below 1.0 or above 3.0.
- Change at most 2 parameters per tuning cycle. Small, incremental adjustments.
- Paper trading: changes are safe, but be conservative and well-reasoned.
"""


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def _launch_bot() -> subprocess.Popen | None:
    global _bot_proc
    print(f"[monitor] Launching: {' '.join(_bot_cmd)}", flush=True)
    try:
        proc = subprocess.Popen(
            _bot_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(_PROJECT_ROOT),
        )
        _bot_proc = proc
        t = threading.Thread(target=_stdout_reader, args=(proc,), daemon=True)
        t.start()
        return proc
    except Exception as exc:
        print(f"[monitor] ERROR launching bot: {exc}", flush=True)
        return None


def _stdout_reader(proc: subprocess.Popen) -> None:
    """Read bot stdout line by line, mirror to our stdout, and fill _log_buffer."""
    assert proc.stdout is not None
    traceback_lines: list[str] = []
    in_traceback = False

    for line in proc.stdout:
        line = line.rstrip("\n")
        _log_buffer.append(line)
        print(f"[bot] {line}", flush=True)

        # Traceback detection
        if _TRACEBACK_START in line:
            in_traceback = True
            traceback_lines = [line]
        elif in_traceback:
            traceback_lines.append(line)
            # Exception lines look like: ExceptionType: message  (no leading whitespace)
            if line and not line.startswith(" ") and not line.startswith("\t") and ":" in line:
                # Likely the final exception line
                in_traceback = False
                tb_text = "\n".join(traceback_lines)
                threading.Thread(
                    target=_handle_traceback, args=(tb_text,), daemon=True
                ).start()
                traceback_lines = []

    print("[monitor] Bot process ended.", flush=True)


# ---------------------------------------------------------------------------
# Bug-fix mode
# ---------------------------------------------------------------------------

def _handle_traceback(traceback_text: str) -> None:
    global _last_fix_time

    now = time.time()
    # Skip if we just applied a fix (cooldown) or another fix is in progress
    if now - _last_fix_time < _COOLDOWN_AFTER_FIX_SEC:
        print("[monitor] Traceback detected but in cooldown — skipping.", flush=True)
        return
    if len(_log_buffer) < _MIN_LINES_BEFORE_BUG_FIX:
        return

    acquired = _fix_lock.acquire(blocking=False)
    if not acquired:
        print("[monitor] Bug-fix already in progress — skipping duplicate.", flush=True)
        return

    try:
        _last_fix_time = now
        print("\n[monitor] === TRACEBACK DETECTED — invoking Claude bug-fix ===", flush=True)
        print(traceback_text, flush=True)

        client = anthropic.Anthropic()
        user_msg = (
            f"The trading bot crashed with this traceback:\n\n```\n{traceback_text}\n```\n\n"
            "Please read the relevant source file(s), fix the bug, and restart the bot."
        )

        runner = client.beta.messages.tool_runner(
            model=MODEL,
            thinking=THINKING,
            max_tokens=MAX_TOKENS,
            system=_BUG_FIX_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            tools=_ALL_TOOLS,
        )
        result = runner.until_done()
        _print_claude_result(result, label="BUG-FIX")
    except Exception as exc:
        print(f"[monitor] ERROR in bug-fix invocation: {exc}", flush=True)
    finally:
        _fix_lock.release()


# ---------------------------------------------------------------------------
# Tuning mode
# ---------------------------------------------------------------------------

def _tuning_loop(interval_min: int) -> None:
    interval_sec = interval_min * 60
    # Wait one full interval before the first tuning run
    time.sleep(interval_sec)
    while True:
        _run_tuning()
        time.sleep(interval_sec)


def _run_tuning() -> None:
    print("\n[monitor] === TUNING CYCLE — invoking Claude strategy tuning ===", flush=True)
    today = date.today().strftime("%Y-%m-%d")
    client = anthropic.Anthropic()
    user_msg = (
        f"Today is {today}. Please analyze today's trades and current config, "
        "then tune the strategy or write recommendations."
    )
    try:
        runner = client.beta.messages.tool_runner(
            model=MODEL,
            thinking=THINKING,
            max_tokens=MAX_TOKENS,
            system=_TUNING_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            tools=_ALL_TOOLS,
        )
        result = runner.until_done()
        _print_claude_result(result, label="TUNING")
    except Exception as exc:
        print(f"[monitor] ERROR in tuning invocation: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_claude_result(result: object, label: str) -> None:
    print(f"\n[monitor] === Claude {label} result ===", flush=True)
    content = getattr(result, "content", [])
    for block in content:
        block_type = getattr(block, "type", "")
        if block_type == "text":
            print(getattr(block, "text", ""), flush=True)
        elif block_type == "thinking":
            summary = getattr(block, "thinking", "") or getattr(block, "summary", "")
            if summary:
                print(f"[thinking] {summary[:300]}...", flush=True)
    print(f"[monitor] === End {label} ===\n", flush=True)


def _sigterm_handler(signum: int, frame: object) -> None:
    print("\n[monitor] Shutting down...", flush=True)
    if _bot_proc is not None:
        _bot_proc.terminate()
    sys.exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global _bot_cmd

    parser = argparse.ArgumentParser(description="Claude AI monitor for the trading bot")
    parser.add_argument(
        "--bot-cmd",
        default=f"{sys.executable} -m {_BOT_MODULE}",
        help="Command to launch the bot (default: python -m bot.main)",
    )
    parser.add_argument(
        "--tune-interval",
        type=int,
        default=30,
        help="Minutes between tuning cycles (default: 30)",
    )
    parser.add_argument(
        "--no-bot",
        action="store_true",
        help="Don't launch the bot — monitor a separately-running process via log tailing",
    )
    args = parser.parse_args()

    _bot_cmd = args.bot_cmd.split()

    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[monitor] WARNING: ANTHROPIC_API_KEY not set — Claude calls will fail.", flush=True)

    # Start tuning thread
    tuning_thread = threading.Thread(
        target=_tuning_loop, args=(args.tune_interval,), daemon=True
    )
    tuning_thread.start()
    print(f"[monitor] Tuning cycle every {args.tune_interval} min.", flush=True)

    if args.no_bot:
        print("[monitor] --no-bot: watching logs only, not launching subprocess.", flush=True)
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            pass
        return

    # Launch bot and keep it alive (restart if it crashes without a traceback)
    while True:
        proc = _launch_bot()
        if proc is None:
            print("[monitor] Failed to launch bot. Retrying in 30s...", flush=True)
            time.sleep(30)
            continue

        proc.wait()
        exit_code = proc.returncode
        print(f"[monitor] Bot exited with code {exit_code}.", flush=True)

        # Give the bug-fix thread time to finish if it was triggered
        time.sleep(5)
        if exit_code == 0:
            print("[monitor] Clean exit. Monitor stopping.", flush=True)
            break
        # Non-zero exit: restart after a short delay
        print("[monitor] Restarting bot in 10s...", flush=True)
        time.sleep(10)


if __name__ == "__main__":
    main()
