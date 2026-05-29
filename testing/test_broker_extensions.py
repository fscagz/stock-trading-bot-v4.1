from unittest.mock import MagicMock, patch
import bot.broker_alpaca as broker


def test_submit_limit_order_buy():
    mock_order = MagicMock()
    mock_order.id = "order-001"
    with patch.object(broker.trading_client, "submit_order", return_value=mock_order) as mock_submit:
        order_id = broker.submit_limit_order("ASTC", 100, "buy", 2.05)
        assert order_id == "order-001"
        call_args = mock_submit.call_args[0][0]
        assert call_args.symbol == "ASTC"
        assert float(call_args.limit_price) == 2.05


def test_submit_stop_order():
    mock_order = MagicMock()
    mock_order.id = "stop-001"
    with patch.object(broker.trading_client, "submit_order", return_value=mock_order) as mock_submit:
        order_id = broker.submit_stop_order("ASTC", 100, 1.70)
        assert order_id == "stop-001"
        call_args = mock_submit.call_args[0][0]
        assert float(call_args.stop_price) == 1.70


def test_cancel_order():
    with patch.object(broker.trading_client, "cancel_order_by_id") as mock_cancel:
        broker.cancel_order("stop-001")
        mock_cancel.assert_called_once_with("stop-001")


def test_get_order_status():
    mock_order = MagicMock()
    mock_order.status = "filled"
    mock_order.filled_avg_price = 2.05
    with patch.object(broker.trading_client, "get_order_by_id", return_value=mock_order):
        order = broker.get_order("order-001")
        assert order.status == "filled"
