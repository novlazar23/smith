# WI-P5-15 — ExecutionLogStore db_path-Wiring (Test Evidence)

Datum: 2026-08-20 · Bezug: offener Punkt aus dem WI-P5-10-Review (Audit-Log
wurde nie persistiert); symmetrisch zu WI-P5-10/11/12 (KillSwitch-Wiring,
Test-Isolation, atomarer State-Write)

## Ziel

`routes.py` erzeugte `execution_log_store = ExecutionLogStore()` **ohne
`db_path`**, und beide `LiveExecutionService`-Instanzen (Paper + Crypto)
wurden **ohne `log_store`** verdrahtet — Audit-Log-Einträge (R5.3) wurden
nie in eine JSON-State-Datei geschrieben und gingen bei jedem
Prozess-Neustart verloren. Zusätzlich schrieb `_save_state` nicht-atomar
via `open(..., "w")` (Crash mid-write → korrupte JSON), und die
conftest-Isolation (WI-P5-11) deckte nur den Kill-Switch-Pfad ab.

Fix: konfiguriertes `db_path` (`Settings.execution_log_state_path`,
Default `data/execution_log.json`), `log_store=`-Wiring in beiden
Services, atomarer `_save_state` (mkstemp + fsync + `os.replace`,
spiegelt `KillSwitch._save_state`, WI-P5-12), neue `db_path`-Property und
`clear()`, conftest-Isolation für den Execution-Log-Pfad.

## Änderung

`src/trading_harness/config.py` — neue Settings-Option:

```diff
     kill_switch_state_path: str = "data/kill_switch.json"
+    execution_log_state_path: str = "data/execution_log.json"
```

`src/trading_harness/api/routes.py` — Store mit `db_path` erzeugt, beide
Services mit `log_store` verdrahtet (Pipeline-Reihenfolge und Semantik
unverändert):

```diff
-execution_log_store = ExecutionLogStore()
+# Persistenter Execution-Audit-Log: State überlebt Prozess-Neustarts (WI-P5-15)
+execution_log_store = ExecutionLogStore(db_path=settings.execution_log_state_path)
```

```diff
     network_policy=network_policy,
     credential_manager=credential_manager,
     config=execution_config,
+    log_store=execution_log_store,
 )
```
(beide Instanzen: `live_execution_service` und `crypto_execution_service`)

`src/trading_harness/services/execution_store.py`:

- neue öffentliche `db_path`-Property (spiegelt `KillSwitch.db_path`),
  gebraucht vom API-Wiring-Test und für Inspektion;
- `_save_state()` neu: `tempfile.mkstemp` im Zielverzeichnis (gleiches
  Dateisystem → atomares `os.replace`), `flush` + `os.fsync`, Modus-Erhaltung
  der State-Datei (`mkstemp` legt 0600 an → `os.chmod`), best-effort
  Tmp-Cleanup bei Fehlern, `OSError` → `logger.warning` (Persistenzfehler
  nicht kritisch, Zustand bleibt im Speicher);
- `_save_state()` in `add()` wird jetzt **unter dem Lock** aufgerufen
  (Snapshot-Konsistenz: die persistierte Liste ist der aktuelle
  In-Memory-Stand);
- neue `clear()`-Methode: leert `_logs` unter dem Lock und persistiert den
  geleerten Zustand — Grundlage der Test-Isolation;
- redundanter lokaler `from datetime import datetime`-Import in
  `_load_state()` entfernt (module-level Import vorhanden).

`tests/conftest.py` — autouse-Fixture `isolated_kill_switch_state`
erweitert: bindet pro Test zusätzlich
`routes.execution_log_store._db_path` auf `tmp_path /
"execution_log.json"` um und ruft `routes.execution_log_store.clear()`
auf, damit kein In-Memory-Log-State an Folgetests weitergegeben wird
(symmetrisch zum Kill-Switch-Teardown). Opt-out über den neuen Marker
`real_execution_log_state` (analog `real_kill_switch_state`, WI-P5-11).

`pyproject.toml` — Marker-Registrierung (1 Zeile, außerhalb der
Datei-Liste des Workitems — siehe Abweichungen):

```diff
     "real_kill_switch_state: Test verifiziert das echte Kill-Switch-Wiring (keine tmp-Umbindung durch conftest-Fixture)",
+    "real_execution_log_state: Test verifiziert das echte ExecutionLogStore-Wiring (keine tmp-Umbindung durch conftest-Fixture)",
```

`tests/test_api_execution.py` — neue Klasse `TestExecutionLogStoreWiring`
(3 Regressionstests):

- `test_execution_log_store_wired_with_state_path`
  (Marker `real_execution_log_state`): `routes.execution_log_store.db_path
  == routes.settings.execution_log_state_path`;
- `test_execution_log_state_survives_process_restart`: API-POST
  `/execution/orders` → Store-Pfad auf tmp-Datei umbinden, Order wird
  REJECTED (`LIVE_EXECUTION_DISABLED`) und im Store geloggt; neue
  `ExecutionLogStore(db_path=…)`-Instanz (simulierter Neustart) lädt den
  Eintrag zurück (`count == 1`, gleicher `decision_id`, Status REJECTED,
  `error == "LIVE_EXECUTION_DISABLED"`);
- `test_api_writes_do_not_touch_real_execution_log_path`:
  Guard-Test — API-Log-Write erzeugt kein `data/execution_log.json` im
  Repository-CWD (existed_before == exists nachher).

## Verifikation

### Baseline (vor Änderung, `main`)

```text
$ uv run pytest -q
754 passed, 1 warning in 52.47s
```

### RED (TDD, Tests vor Implementierung)

```text
$ uv run pytest tests/test_api_execution.py::TestExecutionLogStoreWiring -q
FAILED tests/test_api_execution.py::TestExecutionLogStoreWiring::test_execution_log_store_wired_with_state_path
    E   AttributeError: 'ExecutionLogStore' object has no attribute 'db_path'
    E   Did you mean: '_db_path'?
FAILED tests/test_api_execution.py::TestExecutionLogStoreWiring::test_execution_log_state_survives_process_restart
    E   assert 0 == 1        # neu geladene Instanz: keine persistierten Einträge
2 failed, 1 passed
```

Beide Red-Failures zeigen den Mangel exakt: ohne `db_path`-Property ist
das API-Wiring nicht prüfbar, und ohne `log_store`-Wiring schreibt die
API nichts in den Store (Reload-Count 0). Der Guard-Test
`test_api_writes_do_not_touch_real_execution_log_path` besteht bereits
(der Write-Pfad existierte vor der Änderung gar nicht).

### GREEN (Testklasse nach Implementierung)

```text
$ uv run pytest tests/test_api_execution.py::TestExecutionLogStoreWiring -q
3 passed
```

### Voll-Suite: Test-Ordering-Leak gefunden und behoben (Transparenz)

Der erste Full-Run nach der Implementierung schlug fehl:

```text
$ uv run pytest -q
FAILED tests/test_api_execution.py::TestExecutionLogStoreWiring::test_execution_log_state_survives_process_restart
    E   assert 4 == 1        # In-Memory-Logs aus vorherigen Tests im Singleton
1 failed, 756 passed
```

Root Cause: das `ExecutionLogStore`-API-Singleton akkumuliert
In-Memory-`_logs` über alle Tests; `add()` (jetzt persistierend) und der
Reload-Vergleich wurden dadurch durch Test-Reihenfolge beeinflusst. Fix:
neue `clear()`-Methode + conftest ruft `routes.execution_log_store.clear()`
in der Setup-Phase der Isolation-Fixture auf (frischer, leerer Start pro
Test, symmetrisch zum Kill-Switch-Deactivate-Teardown).

```text
$ uv run pytest "tests/test_api_execution.py::TestCryptoExecutionEndpoints" "tests/test_api_execution.py::TestExecutionLogStoreWiring" -q
10 passed, 1 warning in 0.12s
$ uv run pytest tests/test_api_execution.py -q
27 passed, 1 warning in 0.19s
```

### `make check` (Endzustand)

```text
$ make check
=============================== warnings summary ===============================
.venv/lib/python3.12/site-packages/fastapi/testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
757 passed, 1 warning in 53.95s
uv run ruff check src tests
All checks passed!
uv run mypy src
Success: no issues found in 50 source files
CHECK_EXIT=0
```

757 = Baseline 754 + 3 neue Tests. Warning: bestehende
`StarletteDeprecationWarning` (unverändert).

### Persistenz-Beweis (tmp-Pfad, host `data/` nicht berührt)

```text
$ uv run python  (Skript: mkstemp-Tmp-Dir, ExecutionLogStore-Write, Reload, Tmp-Check)
db_path: /tmp/wi-p5-15-evidence-pjoce9la/execution_log.json
count nach add(): 1
Dateiinhalt: {"logs": [{"id": "exec-1787214802295-0", "decision_id": "dec-evidence-1", "run_id": "run-evidence-1", "symbol": "BTCUSDT", "side": "buy", "status": "FILLED", "order_id": "ord-evidence-1", "error": null, "timestamp": "2026-08-20T08:33:22.295232+00:00"}]}
count nach Reload: 1
Reload-Eintrag: {"decision_id": "dec-evidence-1", "error": null, "id": "exec-1787214802295-0", "order_id": "ord-evidence-1", "run_id": "run-evidence-1", "side": "buy", "status": "FILLED", "symbol": "BTCUSDT", "timestamp": "2026-08-20T08:33:22.295232+00:00"}
Tmp-Rueckstaende: []
PROOF_OK
```

Exakte JSON wird geschrieben, von einer frischen Instanz (simulierter
Prozess-Neustart) 1:1 geladen, UTC-ISO-Timestamp, keine `*.tmp*`-
Rückstände im Zielverzeichnis.

### `data/` vor und nach Testläufen

```text
$ ls -la /root/smith/data/          # nach allen Testläufen
total 12
drwxr-xr-x  2 root root 4096 Aug 20 07:50 .
drwxr-xr-x 19 root root 4096 Aug 20 07:15 ..
-rw-r--r--  1 root root    0 Aug 13 13:22 .gitkeep
-rw-r--r--  1 root root  208 Aug 20 07:50 kill_switch.json
$ sha256sum /root/smith/data/kill_switch.json
1389df52f9a2e125a05a2ee96b13263870a234236506d2487c12cbc06d2383a9  /root/smith/data/kill_switch.json
$ stat -c '%y %a %n' /root/smith/data/kill_switch.json
2026-08-20 07:50:46.656418374 +0000 644 /root/smith/data/kill_switch.json
$ cat /root/smith/data/kill_switch.json
{"enabled": false, "last_toggled_at": 1787212246.656831, "toggle_count": 2, "auto_trigger_enabled": true, "auto_trigger_threshold": 3, "anomaly_streak": 0, "auto_triggered": false, "trigger_reason": "manual"}
```

Keine `data/execution_log.json` auf dem Host; `kill_switch.json`
byte-identisch zum Baseline-Zustand (gleicher sha256 wie vor der Arbeit,
`enabled: false`, `toggle_count: 2`) — die Test-Suite berührt den
echten State-Pfad nicht.

## Abweichungen

- **`pyproject.toml` (1 Zeile):** liegt außerhalb der Datei-Liste des
  Workitems, ist aber für das symmetrische Pattern nötig — WI-P5-11
  registriert `real_kill_switch_state` dort, und ohne Registrierung
  würde pytest `PytestUnknownMarkWarning` für den neuen Marker melden.
- **README Abschnitt 8:** Test-Zähler 754 → 757 (Doku-Aktualität desselben
  Abschnitts, keine semantische Änderung).
- Sonstige Abweichungen: keine. Live-Execution bleibt deaktiviert,
  Kill-Switch-Semantik, Risk-Policy, Whitelists und Limits unverändert.

## Commit

Diese Datei ist Teil des WI-P5-15-Commits (Conventional Commit
`feat(execution): ExecutionLogStore-Persistenz nach data/execution_log.json
(WI-P5-15)`). Da ein Commit seinen eigenen SHA nicht enthalten kann, ist
der SHA über `git log --grep="WI-P5-15" --format='%H'` bzw.
`git log -1 --format='%H'` (am Commit, der diese Datei enthält)
ermittelbar.
