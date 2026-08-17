"""Tests für NetworkPolicy — R5.15–R5.17."""

from trading_harness.services.network_policy import NetworkPolicy, PolicyViolation


class TestNetworkPolicy:
    def test_empty_policy_allows_all(self) -> None:
        policy = NetworkPolicy()
        assert policy.is_allowed("GET", "https://example.com/api") is True
        assert policy.is_allowed("POST", "https://evil.com/inject") is True

    def test_allows_whitelisted_url(self) -> None:
        policy = NetworkPolicy(allowed_patterns=["https://api\\.bybit\\.com/.*"])
        assert policy.is_allowed("POST", "https://api.bybit.com/v5/order/create") is True

    def test_blocks_non_whitelisted_url(self) -> None:
        policy = NetworkPolicy(allowed_patterns=["https://api\\.bybit\\.com/.*"])
        assert policy.is_allowed("POST", "https://evil.com/inject") is False

    def test_blocks_unrelated_exchange(self) -> None:
        policy = NetworkPolicy(
            allowed_patterns=[
                "https://api\\.bybit\\.com/.*",
                "https://api\\.bitget\\.com/.*",
            ]
        )
        assert (
            policy.is_allowed("POST", "https://api.binance.com/order") is False
        )

    def test_allows_wildcard_pattern(self) -> None:
        policy = NetworkPolicy(allowed_patterns=["https://api\\..*/.*"])
        assert (
            policy.is_allowed("POST", "https://api.example.com/v1/orders") is True
        )

    def test_violation_logged(self) -> None:
        policy = NetworkPolicy(allowed_patterns=["https://allowed\\.com/.*"])
        result = policy.is_allowed("POST", "https://evil.com/inject")
        assert result is False
        assert policy.violation_count == 1

    def test_multiple_violations(self) -> None:
        policy = NetworkPolicy(allowed_patterns=["https://allowed\\.com/.*"])
        policy.is_allowed("GET", "https://evil.com/1")
        policy.is_allowed("POST", "https://evil.com/2")
        assert policy.violation_count == 2

    def test_get_violations_returns_list(self) -> None:
        policy = NetworkPolicy(allowed_patterns=["https://allowed\\.com/.*"])
        policy.is_allowed("GET", "https://evil.com/x")
        violations = policy.get_violations()
        assert len(violations) == 1
        v: PolicyViolation = violations[0]
        assert v.method == "GET"
        assert "evil.com" in v.url

    def test_add_allowed_pattern(self) -> None:
        policy = NetworkPolicy(allowed_patterns=["https://old\\.com/.*"])
        assert (
            policy.is_allowed("GET", "https://old.com/path") is True
        )
        assert (
            policy.is_allowed("GET", "https://new.com/path") is False
        )
        policy.add_allowed("https://new\\.com/.*")
        assert (
            policy.is_allowed("GET", "https://new.com/path") is True
        )

    def test_violation_count_property(self) -> None:
        policy = NetworkPolicy()
        assert policy.violation_count == 0
        policy.add_allowed("https://allowed\\.com/.*")
        policy.is_allowed("GET", "https://blocked.com/x")
        assert policy.violation_count == 1

    def test_get_violations_with_limit(self) -> None:
        policy = NetworkPolicy(allowed_patterns=["https://allowed\\.com/.*"])
        for i in range(5):
            policy.is_allowed("GET", f"https://evil{i}.com/x")
        violations = policy.get_violations(limit=2)
        assert len(violations) == 2

    def test_different_methods_same_url(self) -> None:
        policy = NetworkPolicy(allowed_patterns=["https://allowed\\.com/.*"])
        policy.is_allowed("GET", "https://blocked.com/x")
        policy.is_allowed("POST", "https://blocked.com/x")
        policy.is_allowed("DELETE", "https://blocked.com/x")
        assert policy.violation_count == 3