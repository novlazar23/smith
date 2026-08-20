"""Globale Pytest-Fixtures — Test-Isolation für Execution-State.

WI-P5-11 (Review-MINOR-1 von WI-P5-10): API-Tests dürfen den
konfigurierten Kill-Switch-State-Pfad (Default ``data/kill_switch.json``)
nicht beschreiben — sonst startet ``make run`` nach ``make check`` mit
einem durch die Test-Suite aktivierten Kill Switch. Die autouse-Fixture
umbindet ``routes.execution_kill_switch._db_path`` für jeden Test auf
einen tmp-Pfad und deaktiviert danach ein noch aktives Singleton, damit
kein Execution-State an Folgetests weitergegeben wird.

Tests, die das echte API-Wiring verifizieren (z.B.
``test_kill_switch_wired_with_state_path``), setzen den Marker
``real_kill_switch_state`` und laufen ohne Umbindung.
"""

from __future__ import annotations

import pytest

from trading_harness.api import routes


@pytest.fixture(autouse=True)
def isolated_kill_switch_state(tmp_path, monkeypatch, request):
    """Isoliert den Kill-Switch-State des API-Singletons pro Test."""
    if request.node.get_closest_marker("real_kill_switch_state") is not None:
        yield
        return
    monkeypatch.setattr(
        routes.execution_kill_switch,
        "_db_path",
        str(tmp_path / "kill_switch.json"),
    )
    yield
    if routes.execution_kill_switch.is_active():
        routes.execution_kill_switch.deactivate()
