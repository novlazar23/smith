"""Globale Pytest-Fixtures — Test-Isolation für Execution-State.

WI-P5-11 (Review-MINOR-1 von WI-P5-10): API-Tests dürfen den
konfigurierten Kill-Switch-State-Pfad (Default ``data/kill_switch.json``)
nicht beschreiben — sonst startet ``make run`` nach ``make check`` mit
einem durch die Test-Suite aktivierten Kill Switch. Die autouse-Fixture
umbindet ``routes.execution_kill_switch._db_path`` für jeden Test auf
einen tmp-Pfad und deaktiviert danach ein noch aktives Singleton, damit
kein Execution-State an Folgetests weitergegeben wird.

WI-P5-15 (symmetrisch zu WI-P5-11): dieselbe Isolation für den
konfigurierten Execution-Log-State-Pfad (Default
``data/execution_log.json``) — ``routes.execution_log_store._db_path``
wird pro Test auf einen tmp-Pfad umgebunden, damit die Suite die echte
Audit-Log-Datei im Repository nie berührt.

Tests, die das echte API-Wiring verifizieren (z.B.
``test_kill_switch_wired_with_state_path`` bzw.
``test_execution_log_store_wired_with_state_path``), setzen den Marker
``real_kill_switch_state`` bzw. ``real_execution_log_state`` und laufen
für den jeweiligen State ohne Umbindung.
"""

from __future__ import annotations

import os

import pytest

# Never acquire real cross-system leases or start background loops during test collection.
os.environ["STATE_HANDOFF_ENABLED"] = "false"
os.environ["AUTONOMOUS_SHADOW_ENABLED"] = "false"

from trading_harness.api import routes


@pytest.fixture(autouse=True)
def isolated_kill_switch_state(tmp_path, monkeypatch, request):
    """Isoliert den Execution-State (Kill-Switch + Execution-Log) des API-Singletons pro Test."""
    if request.node.get_closest_marker("real_kill_switch_state") is None:
        monkeypatch.setattr(
            routes.execution_kill_switch,
            "_db_path",
            str(tmp_path / "kill_switch.json"),
        )
    if request.node.get_closest_marker("real_execution_log_state") is None:
        monkeypatch.setattr(
            routes.execution_log_store,
            "_db_path",
            str(tmp_path / "execution_log.json"),
        )
        # In-Memory-Logs des Singletons nicht an Folgetests weitergeben
        # (symmetrisch zum Kill-Switch-Teardown): frischer, leerer Start.
        routes.execution_log_store.clear()
    yield
    if request.node.get_closest_marker("real_execution_log_state") is None:
        # Review-NIT (WI-P5-14): Der Marker-Opt-out überspringt den
        # Setup-Clear — der In-Memory-Log-State muss auch im Teardown
        # aufgeräumt werden, damit ein direkt folgender
        # Marker-Opt-out-Test nichts erbt (zukunftssicherungs-relevant).
        # Guard ist zwingend: nur ohne Marker ist der Store auf den
        # tmp-Pfad gebunden; ein Clear im Opt-out-Fall würde in die
        # echte State-Datei persistieren und die Isolation brechen.
        # (Wirkt vor der Monkeypatch-Rücksetzung, also auf dem tmp-Pfad.)
        routes.execution_log_store.clear()
    if routes.execution_kill_switch.is_active():
        routes.execution_kill_switch.deactivate()
