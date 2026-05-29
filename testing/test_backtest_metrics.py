from datetime import datetime, timezone, timedelta
from bot.backtest.backtest_metrics import compute_metrics
from bot.intraday.types import TradeRecord


def _record(pnl: float, exit_reason: str = "hard_stop", hold_minutes: int = 30) -> TradeRecord:
    entry = datetime(2025, 3, 10, 14, 0, tzinfo=timezone.utc)
    exit_ = entry + timedelta(minutes=hold_minutes)
    return TradeRecord(
        ticker="TEST",
        direction="long",
        entry_time=entry,
        entry_price=10.0,
        shares=100,
        stop_price=9.0,
        target_price=12.0,
        signals=["momentum"],
        sector="Unknown",
        regime="",
        portfolio_heat_at_entry=0.01,
        expected_slippage_pct=0.001,
        exit_time=exit_,
        exit_price=10.0 + pnl / 100,
        pnl=pnl,
        exit_reason=exit_reason,
    )


def test_empty_trades():
    result = compute_metrics([], 10000.0)
    assert result["total_trades"] == 0
    assert result["win_rate"] == 0.0
    assert result["total_pnl"] == 0.0


def test_win_rate():
    trades = [_record(100.0), _record(50.0), _record(-30.0), _record(-20.0)]
    result = compute_metrics(trades, 10000.0)
    assert result["total_trades"] == 4
    assert result["win_rate"] == 0.5


def test_total_pnl():
    trades = [_record(100.0), _record(-40.0)]
    result = compute_metrics(trades, 10000.0)
    assert result["total_pnl"] == 60.0


def test_avg_winner_and_loser():
    trades = [_record(100.0), _record(200.0), _record(-50.0), _record(-100.0)]
    result = compute_metrics(trades, 10000.0)
    assert result["avg_winner"] == 150.0
    assert result["avg_loser"] == -75.0


def test_max_drawdown():
    # equity: 10000 +100 +200 -300 -100 → peaks at 10300, troughs at 9900 → drawdown=400
    trades = [_record(100.0), _record(200.0), _record(-300.0), _record(-100.0)]
    result = compute_metrics(trades, 10000.0)
    assert result["max_drawdown"] == 400.0


def test_avg_hold_minutes():
    trades = [_record(10.0, hold_minutes=30), _record(-10.0, hold_minutes=60)]
    result = compute_metrics(trades, 10000.0)
    assert result["avg_hold_minutes"] == 45.0


def test_exit_reasons():
    trades = [
        _record(10.0, exit_reason="hard_stop"),
        _record(-10.0, exit_reason="hard_stop"),
        _record(20.0, exit_reason="eod"),
    ]
    result = compute_metrics(trades, 10000.0)
    assert result["exit_reasons"]["hard_stop"] == 2
    assert result["exit_reasons"]["eod"] == 1
