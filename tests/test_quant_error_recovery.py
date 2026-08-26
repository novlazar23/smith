"""Tests für Error Recovery."""
from __future__ import annotations

from trading_harness.quant.error_recovery import ErrorRecovery, RetryConfig


class TestErrorRecovery:
    def test_with_retry_success(self):
        recovery = ErrorRecovery(RetryConfig(max_retries=3, base_delay=0.01))
        call_count = 0
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("fail")
            return "ok"
        result = recovery.with_retry(func, operation_name="test")
        assert result.success
        assert result.value == "ok"
        assert result.retries == 1

    def test_with_retry_all_fail(self):
        recovery = ErrorRecovery(RetryConfig(max_retries=2, base_delay=0.01))
        def func():
            raise ValueError("always fail")
        result = recovery.with_retry(func)
        assert not result.success
        assert result.retries == 2

    def test_with_fallback_success(self):
        recovery = ErrorRecovery()
        result = recovery.with_fallback(lambda: "ok", "fallback")
        assert result == "ok"

    def test_with_fallback_failure(self):
        recovery = ErrorRecovery()
        result = recovery.with_fallback(lambda: (_ for _ in ()).throw(ValueError("fail")), "fallback")
        assert result == "fallback"

    def test_safe_execute_success(self):
        recovery = ErrorRecovery()
        result = recovery.safe_execute(lambda: 42, default=0)
        assert result == 42

    def test_safe_execute_failure(self):
        recovery = ErrorRecovery()
        result = recovery.safe_execute(lambda: (_ for _ in ()).throw(ValueError("fail")), default=0)
        assert result == 0

    def test_register_fallback(self):
        recovery = ErrorRecovery()
        recovery.register_fallback("key", "value")
        assert recovery.get_fallback("key") == "value"
        assert recovery.get_fallback("missing", "default") == "default"

    def test_validate_and_fix(self):
        recovery = ErrorRecovery()
        schema = {"name": {}, "count": {"default": 0}}
        result = recovery.validate_and_fix({"name": "test"}, schema)
        assert result == {"name": "test", "count": 0}

    def test_deterministic(self):
        recovery = ErrorRecovery(RetryConfig(max_retries=0, base_delay=0.01))
        r1 = recovery.safe_execute(lambda: 1)
        r2 = recovery.safe_execute(lambda: 1)
        assert r1 == r2
