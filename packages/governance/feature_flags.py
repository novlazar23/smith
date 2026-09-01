"""Feature Flags — Zentrale Verwaltung von Feature-Flags.

Sicherheitskritische Feature-Flags werden durch eine Singleton-Klasse
verwaltet. Das Flag `live_trading_enabled` ist standardmäßig deaktiviert
und muss explizit in Produktion freigegeben werden.
"""

from __future__ import annotations

import logging
import os

from packages.governance.audit import AuditTrail

logger = logging.getLogger(__name__)


class FeatureFlags:
    """Singleton zur Verwaltung von Feature-Flags mit Audit-Trail.

    - In Entwicklung und Staging ist live_trading_enabled immer False.
    - In Produktion muss das Flag explizit auf True gesetzt werden.
    """

    _instance: FeatureFlags | None = None

    def __new__(cls) -> FeatureFlags:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return
        self._initialized: bool = True
        self._flags: dict[str, bool] = {
            "live_trading_enabled": False,
            "batch_processing_enabled": False,
        }
        self._audit = AuditTrail()
        self._environment: str = os.environ.get(
            "APP_ENV", os.environ.get("ENV", "development")
        )

    def is_enabled(self, flag: str, environment: str | None = None) -> bool:
        """Prüft, ob ein Feature-Flag aktiviert ist.

        In Entwicklung/Staging wird live_trading_enabled immer False
        zurückgegeben (Safety-first). In Produktion wird das Flags-Dict
        abgefragt.
        """
        env = environment or self._environment

        if flag == "live_trading_enabled" and env in (
            "development",
            "staging",
        ):
            return False

        return self._flags.get(flag, False)

    def set_flag(self, flag: str, *, enabled: bool) -> None:
        """Setzt ein Feature-Flag und protokolliert die Änderung."""
        old_value = self._flags.get(flag, False)
        if old_value == enabled:
            return
        self._flags[flag] = enabled
        logger.info("Feature flag %s changed: %s -> %s", flag, old_value, enabled)
        self._audit.log_decision(
            agent_id="feature-flags",
            decision=f"{flag}={enabled}",
            actor="system",
            details={
                "event": "flag_change",
                "flag": flag,
                "old_value": old_value,
                "new_value": enabled,
            },
        )

    def get_all_flags(self) -> dict[str, bool]:
        """Gibt eine Kopie aller aktuellen Flags zurück."""
        return dict(self._flags)

    def log_flag_change(
        self,
        flag: str,
        *,
        old_value: bool,
        new_value: bool,
        actor: str,
    ) -> None:
        """Protokolliert eine Flag-Änderung im Audit-Trail."""
        self._audit.log_decision(
            agent_id="feature-flags",
            decision=f"{flag}: {old_value} -> {new_value}",
            actor=actor,
            details={
                "event": "flag_change",
                "flag": flag,
                "old_value": old_value,
                "new_value": new_value,
                "actor": actor,
            },
        )
        logger.info(
            "Audit: flag %s changed %s -> %s by %s",
            flag,
            old_value,
            new_value,
            actor,
        )

    @property
    def environment(self) -> str:
        return self._environment


# Module-Level Singleton-Instanz
feature_flags: FeatureFlags = FeatureFlags()
