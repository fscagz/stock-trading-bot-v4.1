import csv
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from bot.intraday.types import TradeRecord
from bot.trade_logger import TradeLogger


def _record(ticker="ASTC", exit_reason="hard_stop", pnl=-30.0):
    entry = datetime(2026, 5, 29, 14, 0, tzinfo=timezone.utc)
    exit_ = datetime(2026, 5, 29, 14, 30, tzinfo=timezone.utc)
    return TradeRecord(
        ticker=ticker,
        direction="long",
        entry_time=entry,
        entry_price=2.00,
        shares=100,
        stop_price=1.70,
        target_price=2.60,
        signals=["momentum"],
        sector="Unknown",
        regime="",
        portfolio_heat_at_entry=0.005,
        expected_slippage_pct=0.0005,
        exit_time=exit_,
        exit_price=1.70,
        pnl=pnl,
        exit_reason=exit_reason,
    )


def test_log_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TradeLogger(log_dir=tmpdir)
        logger.log(_record())
        files = list(Path(tmpdir).glob("trades_*.csv"))
        assert len(files) == 1


def test_log_filename_uses_entry_date():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TradeLogger(log_dir=tmpdir)
        logger.log(_record())
        files = list(Path(tmpdir).glob("trades_*.csv"))
        assert files[0].name == "trades_2026-05-29.csv"


def test_log_writes_header_and_row():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TradeLogger(log_dir=tmpdir)
        logger.log(_record())
        path = Path(tmpdir) / "trades_2026-05-29.csv"
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 1
        assert rows[0]["ticker"] == "ASTC"
        assert rows[0]["exit_reason"] == "hard_stop"
        assert rows[0]["pnl"] == "-30.0"


def test_log_appends_second_trade():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TradeLogger(log_dir=tmpdir)
        logger.log(_record(ticker="ASTC", pnl=-30.0))
        logger.log(_record(ticker="VMAR", pnl=50.0))
        path = Path(tmpdir) / "trades_2026-05-29.csv"
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 2
        assert rows[1]["ticker"] == "VMAR"


def test_log_does_not_write_header_twice():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TradeLogger(log_dir=tmpdir)
        logger.log(_record(ticker="ASTC"))
        logger.log(_record(ticker="VMAR"))
        path = Path(tmpdir) / "trades_2026-05-29.csv"
        lines = path.read_text().splitlines()
        header_count = sum(1 for l in lines if l.startswith("ticker"))
        assert header_count == 1
