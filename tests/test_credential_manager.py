"""Tests für CredentialManager — R5.18–R5.19."""

import os

from trading_harness.services.credential_manager import CredentialManager


class TestCredentialManager:
    def test_register_creates_ref(self) -> None:
        mgr = CredentialManager()
        ref = mgr.register("BYBIT_API_KEY", "env")
        assert ref.key == "BYBIT_API_KEY"
        assert ref.source == "env"

    def test_get_returns_env_value(self) -> None:
        os.environ["TEST_CRED_KEY"] = "secret_value_123"
        mgr = CredentialManager()
        mgr.register("TEST_CRED_KEY")
        assert mgr.get("TEST_CRED_KEY") == "secret_value_123"
        del os.environ["TEST_CRED_KEY"]

    def test_get_returns_none_when_missing(self) -> None:
        mgr = CredentialManager()
        mgr.register("NONEXISTENT_KEY")
        assert mgr.get("NONEXISTENT_KEY") is None

    def test_is_configured_returns_true(self) -> None:
        os.environ["CRED_TEST"] = "some_value"
        mgr = CredentialManager()
        mgr.register("CRED_TEST")
        assert mgr.is_configured("CRED_TEST") is True
        del os.environ["CRED_TEST"]

    def test_is_configured_returns_false_when_empty(self) -> None:
        mgr = CredentialManager()
        mgr.register("EMPTY_KEY")
        assert mgr.is_configured("EMPTY_KEY") is False

    def test_summary_no_raw_values(self) -> None:
        os.environ["SUMMARY_TEST"] = "super_secret_456"
        mgr = CredentialManager()
        mgr.register("SUMMARY_TEST")
        summary = mgr.summary()
        assert "SUMMARY_TEST" in summary["configured"]
        assert "super_secret_456" not in str(summary)
        del os.environ["SUMMARY_TEST"]

    def test_multiple_credentials(self) -> None:
        os.environ["KEY_A"] = "value_a"
        os.environ["KEY_B"] = "value_b"
        mgr = CredentialManager()
        mgr.register("KEY_A")
        mgr.register("KEY_B")
        assert mgr.get("KEY_A") == "value_a"
        assert mgr.get("KEY_B") == "value_b"
        assert len(mgr.configured_keys) == 2
        del os.environ["KEY_A"]
        del os.environ["KEY_B"]

    def test_clear_cache(self) -> None:
        os.environ["CACHE_TEST"] = "cached_val"
        mgr = CredentialManager()
        mgr.register("CACHE_TEST")
        _ = mgr.get("CACHE_TEST")
        assert "CACHE_TEST" in mgr._cache
        mgr.clear_cache()
        assert "CACHE_TEST" not in mgr._cache
        del os.environ["CACHE_TEST"]

    def test_get_caches_value(self) -> None:
        os.environ["CACHE_TEST2"] = "cache_val"
        mgr = CredentialManager()
        mgr.register("CACHE_TEST2")
        v1 = mgr.get("CACHE_TEST2")
        v2 = mgr.get("CACHE_TEST2")
        assert v1 == v2
        assert len(mgr._cache) == 1
        del os.environ["CACHE_TEST2"]

    def test_registered_keys_list(self) -> None:
        mgr = CredentialManager()
        mgr.register("KEY_1")
        mgr.register("KEY_2")
        mgr.register("KEY_3")
        assert len(mgr._refs) == 3

    def test_configured_keys_reflect_env(self) -> None:
        os.environ["ENV_1"] = "val1"
        mgr = CredentialManager()
        mgr.register("ENV_1")
        mgr.register("ENV_NOT_SET")
        assert "ENV_1" in mgr.configured_keys
        assert "ENV_NOT_SET" not in mgr.configured_keys
        del os.environ["ENV_1"]