from __future__ import annotations
from datetime import datetime, timedelta
from typing import Tuple

from bot.intraday.config import IntradayConfig
from bot.intraday.risk.portfolio import PortfolioState


class KillSwitch:
    """Checks kill-switch and cooldown conditions; mutates PortfolioState as needed.

    Kill conditions (halt all trading):
      1. Daily P&L drawdown exceeds max_daily_drawdown
      2. Session slippage > slippage_kill_multiple × expected

    Cooldown condition (30-min pause, not a full halt):
      3. consecutive_losses >= consecutive_loss_trigger
    """

    def __init__(self, config: IntradayConfig) -> None:
        self._cfg = config

    def check(self, state: PortfolioState, now: datetime) -> Tuple[bool, str]:
        """Return (kill_triggered, reason). Mutates state if triggered."""
        # 1. Daily drawdown
        drawdown = state.daily_pnl_pct()
        if drawdown <= -self._cfg.max_daily_drawdown:
            state.kill_switch_active = True
            return True, f"daily_drawdown: {drawdown:.2%}"

        # 2. Slippage spike
        if (state.session_slippage_expected > 0 and
                state.session_slippage_actual / state.session_slippage_expected
                >= self._cfg.slippage_kill_multiple):
            state.kill_switch_active = True
            ratio = state.session_slippage_actual / state.session_slippage_expected
            return True, f"slippage_spike: ratio={ratio:.2f}x"

        # 3. Consecutive loss cooldown (not a kill switch — just a pause)
        if (state.consecutive_losses >= self._cfg.consecutive_loss_trigger
                and not state.in_cooldown(now)):
            cooldown_end = now + timedelta(minutes=self._cfg.consecutive_loss_cooldown_minutes)
            state.cooldown_until = cooldown_end

        return False, ""
