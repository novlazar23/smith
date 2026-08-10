from trading_harness.services.execution_gateway import ExecutionGateway


def test_live_execution_disabled_by_default():
    gateway = ExecutionGateway(live_enabled=False)
    assert gateway.submit("d1")["reason"] == "LIVE_EXECUTION_DISABLED"


def test_no_adapter_even_when_enabled():
    gateway = ExecutionGateway(live_enabled=True)
    result = gateway.submit("d1")
    assert result["status"] == "REJECTED"
    assert result["reason"] == "NO_EXCHANGE_ADAPTER_IMPLEMENTED"
