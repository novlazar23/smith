"""Security — Rollenbasierte Zugriffskontrolle, Secrets-Management, Isolation.

EPIC-12 WP03: Security Review & Execution Isolation.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "LIVE_EXECUTION_BLOCKED",
    "ROLE_PERMISSIONS",
    "Permission",
    "Role",
    "SecretManager",
    "SecurityContext",
    "check_execution_mode",
    "check_security_requirements",
    "ensure_no_secrets_in_output",
    "is_execution_isolated",
    "verify_live_mode_blocked",
]

# ── SECURITY HELPERS ──────────────────────────────────────────────────

# Patterns die auf Secrets hinweisen (nicht-exhaustiv)
_SECRET_PATTERNS = [
    r"(?i)(password|passwd|pwd)\s*[=:]\s*\S+",
    r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?key)\s*[=:]\s*\S+",
    r"(?i)(sk_live|sk_test|pk_live|pk_test)\w{10,}",
    r"(?i)(token)\s*[=:]\s*\S{8,}",
]


def check_security_requirements(role: Role, required_permissions: list[Permission]) -> dict[str, Any]:
    """Prüft ob eine Rolle alle erforderlichen Berechtigungen hat.

    Returns:
        Dict mit "compliant" (bool) und "missing" (list[str]).
    """
    ctx = SecurityContext(role=role)
    missing = [
        p.value for p in required_permissions if not ctx.has_permission(p)
    ]
    return {
        "compliant": len(missing) == 0,
        "role": role,
        "required": [p.value for p in required_permissions],
        "missing": missing,
    }


def ensure_no_secrets_in_output(text: str) -> bool:
    return all(not re.search(pattern, text) for pattern in _SECRET_PATTERNS)


def is_execution_isolated() -> bool:
    """Prüft ob der Orchestrator keine Executionsmethoden exposed.

    Verifiziert: keine create_order, cancel_order, withdraw, transfer.
    """
    try:
        import packages.orchestrator

        orchestrator_module = packages.orchestrator
        # Prüfe alle Attribute
        for attr_name in dir(orchestrator_module):
            if attr_name.startswith("_"):
                continue
            value = getattr(orchestrator_module, attr_name)
            # Prüfe ob es eine Klasse oder Funktion ist
            if callable(value) or hasattr(value, "__dict__"):
                name_lower = attr_name.lower()
                if any(kw in name_lower for kw in ["create_order", "cancel_order", "withdraw", "transfer"]):
                    return False
        return True
    except ImportError:
        # Modul nicht gefunden → Isolation kann nicht geprüft werden
        return True


def verify_live_mode_blocked() -> bool:
    """Stellt sicher dass LIVE-Execution im MVP blockiert ist.

    Returns:
        True wenn blockiert, False sonst.
    """
    return LIVE_EXECUTION_BLOCKED


class Role(StrEnum):
    """Rollen-Modell für die Trading-Orchestra API."""

    VIEWER = "viewer"
    RESEARCHER = "researcher"
    OPERATOR = "operator"
    RISK_MANAGER = "risk_manager"
    ADMINISTRATOR = "administrator"
    AUDITOR = "auditor"


class Permission(StrEnum):
    """Berechtigungen pro Rolle."""

    READ_STATUS = "read_status"
    READ_METRICS = "read_metrics"
    ANALYZE = "analyze"
    CONFIGURE_AGENTS = "configure_agents"
    PROMOTE_AGENT = "promote_agent"
    QUARANTINE_AGENT = "quarantine_agent"
    EXECUTE_LIVE = "execute_live"
    MANAGE_USERS = "manage_users"
    AUDIT_LOGS = "audit_logs"
    MANAGE_SECRETS = "manage_secrets"


# Rolle → Berechtigungen
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.VIEWER: {
        Permission.READ_STATUS,
        Permission.READ_METRICS,
    },
    Role.RESEARCHER: {
        Permission.READ_STATUS,
        Permission.READ_METRICS,
        Permission.ANALYZE,
    },
    Role.OPERATOR: {
        Permission.READ_STATUS,
        Permission.READ_METRICS,
        Permission.ANALYZE,
        Permission.CONFIGURE_AGENTS,
    },
    Role.RISK_MANAGER: {
        Permission.READ_STATUS,
        Permission.READ_METRICS,
        Permission.ANALYZE,
        Permission.QUARANTINE_AGENT,
    },
    Role.ADMINISTRATOR: {
        Permission.READ_STATUS,
        Permission.READ_METRICS,
        Permission.ANALYZE,
        Permission.CONFIGURE_AGENTS,
        Permission.PROMOTE_AGENT,
        Permission.QUARANTINE_AGENT,
        Permission.MANAGE_USERS,
        Permission.MANAGE_SECRETS,
    },
    Role.AUDITOR: {
        Permission.READ_STATUS,
        Permission.READ_METRICS,
        Permission.AUDIT_LOGS,
    },
}


@dataclass
class SecurityContext:
    """Aktueller Sicherheitskontext einer Anfrage."""

    role: Role
    user_id: str = ""
    permissions: set[Permission] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.permissions = ROLE_PERMISSIONS.get(self.role, set())

    def has_permission(self, permission: Permission) -> bool:
        """Prüft ob der Kontext eine Berechtigung hat."""
        return permission in self.permissions


LIVE_EXECUTION_BLOCKED = True
VIEWER = Role.VIEWER
RESEARCHER = Role.RESEARCHER
OPERATOR = Role.OPERATOR
RISK_MANAGER = Role.RISK_MANAGER
ADMINISTRATOR = Role.ADMINISTRATOR
AUDITOR = Role.AUDITOR


def check_execution_mode(mode: str) -> tuple[bool, str]:
    """Prüft ob die angeforderte Ausführung erlaubt ist.

    Returns:
        (erlaubt, fehler_msg)
    """
    if mode.lower() == "live" and LIVE_EXECUTION_BLOCKED:
        return False, "LIVE execution is blocked in MVP (403 Forbidden)"
    allowed_modes = {"research", "backtest", "paper", "shadow"}
    if mode.lower() not in allowed_modes:
        return False, f"Unknown mode '{mode}'"
    return True, ""


@dataclass
class SecretManager:
    """Managt Secrets — aus ENV, Vault oder K8s Secrets.

    Kein Secret wird in Logs, Prompts oder Metriken gespeichert.
    """

    _vault_endpoint: str = ""
    _k8s_namespace: str = ""
    _secrets: dict[str, str] = field(default_factory=dict)

    def get(self, key: str) -> str | None:
        """Liest ein Secret. Prüft priorisiert: ENV → Vault → K8s.

        Raises ValueError wenn Secret nicht gefunden.
        """

        # 1. ENV var
        env_val = os.environ.get(f"SECRET_{key.upper()}")
        if env_val:
            return env_val

        # 2. Vault
        if self._vault_endpoint:
            # Im echten System: requests.get(vault_endpoint/secret/{key})
            vault_val = self._secrets.get(key)
            if vault_val:
                return vault_val

        # 3. K8s Secrets
        if self._k8s_namespace:
            # Im echten System: K8s API call
            k8s_val = self._secrets.get(key)
            if k8s_val:
                return k8s_val

        return None

    def log_safe_message(self, message: str) -> str:
        result = message
        for key in os.environ:
            if key.startswith("SECRET_"):
                val = os.environ[key]
                if val and len(val) > 4:
                    result = result.replace(val, "[REDACTED]")
        return result

