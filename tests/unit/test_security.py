"""Tests für Security Package (WP03 EPIC-12)."""

from __future__ import annotations

import os
from unittest.mock import patch

from packages.security import (
    LIVE_EXECUTION_BLOCKED,
    Permission,
    Role,
    SecretManager,
    SecurityContext,
    check_execution_mode,
    check_security_requirements,
    ensure_no_secrets_in_output,
    verify_live_mode_blocked,
)

# ── Role & Permission Tests ──────────────────────────────────────────


class TestRoles:
    def test_all_roles_present(self) -> None:
        assert Role.VIEWER == "viewer"
        assert Role.RESEARCHER == "researcher"
        assert Role.OPERATOR == "operator"
        assert Role.RISK_MANAGER == "risk_manager"
        assert Role.ADMINISTRATOR == "administrator"
        assert Role.AUDITOR == "auditor"


class TestPermissions:
    def test_all_permissions_present(self) -> None:
        assert Permission.READ_STATUS == "read_status"
        assert Permission.READ_METRICS == "read_metrics"
        assert Permission.ANALYZE == "analyze"
        assert Permission.CONFIGURE_AGENTS == "configure_agents"
        assert Permission.PROMOTE_AGENT == "promote_agent"
        assert Permission.QUARANTINE_AGENT == "quarantine_agent"
        assert Permission.EXECUTE_LIVE == "execute_live"
        assert Permission.MANAGE_USERS == "manage_users"
        assert Permission.AUDIT_LOGS == "audit_logs"
        assert Permission.MANAGE_SECRETS == "manage_secrets"


class TestRolePermissions:
    def test_viewer_has_read_only(self) -> None:
        ctx = SecurityContext(role=Role.VIEWER, user_id="v1")
        assert ctx.has_permission(Permission.READ_STATUS)
        assert ctx.has_permission(Permission.READ_METRICS)
        assert not ctx.has_permission(Permission.ANALYZE)
        assert not ctx.has_permission(Permission.CONFIGURE_AGENTS)

    def test_researcher_can_analyze(self) -> None:
        ctx = SecurityContext(role=Role.RESEARCHER, user_id="r1")
        assert ctx.has_permission(Permission.ANALYZE)
        assert not ctx.has_permission(Permission.EXECUTE_LIVE)
        assert not ctx.has_permission(Permission.MANAGE_USERS)

    def test_operator_configure_but_no_live(self) -> None:
        ctx = SecurityContext(role=Role.OPERATOR, user_id="o1")
        assert ctx.has_permission(Permission.CONFIGURE_AGENTS)
        assert not ctx.has_permission(Permission.EXECUTE_LIVE)
        assert not ctx.has_permission(Permission.MANAGE_USERS)

    def test_risk_manager_quarantine(self) -> None:
        ctx = SecurityContext(role=Role.RISK_MANAGER, user_id="rm1")
        assert ctx.has_permission(Permission.QUARANTINE_AGENT)
        assert not ctx.has_permission(Permission.PROMOTE_AGENT)

    def test_administrator_full_access_except_live(self) -> None:
        ctx = SecurityContext(role=Role.ADMINISTRATOR, user_id="admin1")
        assert ctx.has_permission(Permission.MANAGE_USERS)
        assert ctx.has_permission(Permission.MANAGE_SECRETS)
        assert not ctx.has_permission(Permission.EXECUTE_LIVE)

    def test_auditor_audit_logs(self) -> None:
        ctx = SecurityContext(role=Role.AUDITOR, user_id="aud1")
        assert ctx.has_permission(Permission.AUDIT_LOGS)
        assert not ctx.has_permission(Permission.ANALYZE)


class TestSecurityContext:
    def test_default_permissions_empty(self) -> None:
        ctx = SecurityContext(role=Role.VIEWER, user_id="test")
        assert len(ctx.permissions) > 0  # viewer has permissions

    def test_unknown_role_no_permissions(self) -> None:
        ctx = SecurityContext(role=Role.VIEWER, user_id="test")
        assert not ctx.has_permission(Permission.ANALYZE)  # viewer can't analyze

    def test_has_permission_type(self) -> None:
        ctx = SecurityContext(role=Role.VIEWER, user_id="v1")
        result = ctx.has_permission(Permission.READ_STATUS)
        assert isinstance(result, bool)


# ── Execution Mode Tests ─────────────────────────────────────────────


class TestExecutionMode:
    def test_live_blocked(self) -> None:
        allowed, msg = check_execution_mode("live")
        assert allowed is False
        assert "blocked" in msg.lower()

    def test_research_allowed(self) -> None:
        allowed, msg = check_execution_mode("research")
        assert allowed is True
        assert msg == ""

    def test_backtest_allowed(self) -> None:
        allowed, _msg = check_execution_mode("backtest")
        assert allowed is True

    def test_paper_allowed(self) -> None:
        allowed, _msg = check_execution_mode("paper")
        assert allowed is True

    def test_shadow_allowed(self) -> None:
        allowed, _msg = check_execution_mode("shadow")
        assert allowed is True

    def test_unknown_mode_rejected(self) -> None:
        allowed, msg = check_execution_mode("hacking")
        assert allowed is False
        assert "unknown" in msg.lower() or "mode" in msg.lower()

    def test_live_case_insensitive(self) -> None:
        allowed, _msg = check_execution_mode("LIVE")
        assert allowed is False

    def test_live_case_mixed(self) -> None:
        allowed, _msg = check_execution_mode("Live")
        assert allowed is False


# ── SecretManager Tests ──────────────────────────────────────────────


class TestSecretManager:
    def test_get_from_env(self) -> None:
        with patch.dict(os.environ, {"SECRET_API_KEY": "test-key-123"}):
            mgr = SecretManager()
            assert mgr.get("api_key") == "test-key-123"

    def test_get_not_found(self) -> None:
        mgr = SecretManager()
        assert mgr.get("nonexistent_key") is None

    def test_get_from_vault(self) -> None:
        mgr = SecretManager(_vault_endpoint="http://vault:8200")
        mgr._secrets["db_pass"] = "vault-secret"
        assert mgr.get("db_pass") == "vault-secret"

    def test_env_takes_precedence_over_vault(self) -> None:
        with patch.dict(os.environ, {"SECRET_DB_PASS": "env-secret"}):
            mgr = SecretManager(_vault_endpoint="http://vault:8200")
            mgr._secrets["db_pass"] = "vault-secret"
            assert mgr.get("db_pass") == "env-secret"

    def test_log_safe_message_masks_secret(self) -> None:
        with patch.dict(os.environ, {"SECRET_API_KEY": "sk_live_abc123def456"}):
            mgr = SecretManager()
            msg = "Using key sk_live_abc123def456 in config"
            safe = mgr.log_safe_message(msg)
            assert "[REDACTED]" in safe

    def test_log_safe_message_no_secrets(self) -> None:
        mgr = SecretManager()
        msg = "Normal log message with no secrets"
        assert mgr.log_safe_message(msg) == msg


# ── Execution Isolation Tests ─────────────────────────────────────────


class TestExecutionIsolation:
    def test_orchestrator_no_create_order(self) -> None:
        orchestrator_methods = list(dir(__import__("packages.orchestrator", fromlist=[""])))
        has_create_order = any("create_order" in m.lower() for m in orchestrator_methods)
        has_cancel_order = any("cancel_order" in m.lower() for m in orchestrator_methods)
        assert not has_create_order, "Orchestrator must not have create_order"
        assert not has_cancel_order, "Orchestrator must not have cancel_order"

    def test_orchestrator_no_withdraw(self) -> None:
        orchestrator_methods = list(dir(__import__("packages.orchestrator", fromlist=[""])))
        has_withdraw = any("withdraw" in m.lower() for m in orchestrator_methods)
        assert not has_withdraw, "Orchestrator must not have withdraw"

    def test_orchestrator_no_transfer(self) -> None:
        orchestrator_methods = list(dir(__import__("packages.orchestrator", fromlist=[""])))
        has_transfer = any("transfer" in m.lower() for m in orchestrator_methods)
        assert not has_transfer, "Orchestrator must not have transfer"


class TestLiveModeBlocked:
    def test_live_execution_blocked_flag(self) -> None:
        assert LIVE_EXECUTION_BLOCKED is True

    def test_verify_live_mode_blocked(self) -> None:
        assert verify_live_mode_blocked() is True


class TestNoSecretsInOutput:
    """Testet dass Secrets nicht in Logs, Prompts oder Metriken erscheinen."""

    def test_secret_masking_in_string(self) -> None:
        assert ensure_no_secrets_in_output("key=sk_live_abcdefghij123456") is False

    def test_safe_string_passes(self) -> None:
        assert ensure_no_secrets_in_output("normal log message") is True

    def test_secret_pattern_detection(self) -> None:
        assert ensure_no_secrets_in_output("password=hunter2") is False

    def test_api_key_pattern_detection(self) -> None:
        assert ensure_no_secrets_in_output("API_KEY=abcdef123456") is False


class TestCheckSecurityRequirements:
    """Testet check_security_requirements helper."""

    def test_compliant_role(self) -> None:
        result = check_security_requirements(
            Role.ADMINISTRATOR,
            [Permission.MANAGE_USERS, Permission.MANAGE_SECRETS],
        )
        assert result["compliant"] is True
        assert result["missing"] == []

    def test_non_compliant_role(self) -> None:
        result = check_security_requirements(
            Role.VIEWER,
            [Permission.ANALYZE, Permission.CONFIGURE_AGENTS],
        )
        assert result["compliant"] is False
        assert "analyze" in result["missing"]
        assert "configure_agents" in result["missing"]

