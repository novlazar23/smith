# WI-P5-14 — NIT-Bundle P5-10/P5-11 (Test-Evidence)

Datum: 2026-08-20 · Bezug: nicht blockierende NITs aus dem WI-P5-10-Review
(Review-ID 2) und dem WI-P5-11-Review (Review-ID 3) zur
Execution-State-Test-Isolation

## Ziel

Drei NITs schließen:

- **A**: conftest-Teardown räumt den In-Memory-Log-State des API-
  Singletons auf, auch wenn der direkt folgende Test per
  `real_execution_log_state`-Marker optiert — der Marker überspringt
  heute Setup und Teardown der Log-Isolation (zukunftssicherungs-
  relevant, da der heutige einzige Marker-Test rein lesend ist; ein
  schreibender Marker-Test hätte den Defekt ausgelöst).
- **B**: Das harte `assert response.status_code == 200` im
  `finally`-Cleanup-Block von
  `test_kill_switch_toggle_via_api_leaves_real_state_file_untouched`
  (`tests/test_api_security.py`) kann die Original-Exception des Tests
  maskieren → best-effort-Cleanup ohne hartes Assert.
- **C**: Pin-Test für den Fail-Open-Fallback von
  `KillSwitch._load_state` bei extern korrumpiertem State-File
  ("Fallback: Startzustand verwenden") — wird gepinnt, nicht geändert
  (Fail-Closed wäre eine Sicherheitsgrenzen-Änderung und erfordert
  explizite Freigabe).

## Änderung

`tests/conftest.py` (NIT A) — Teardown der Autouse-Fixture
`isolated_kill_switch_state`, nach dem `yield` und vor dem
Kill-Switch-`deactivate()`:

```python
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
```

Setup-Clear (Session-Start-Schutz) und Kill-Switch-Teardown bleiben
unverändert.

`tests/test_api_security.py` (NIT B):

```diff
             finally:
-                # Singleton nicht im aktiven Zustand für Folgetests hinterlassen
+                # Best-effort-Cleanup ohne hartes Assert — ein
+                # Cleanup-Fehler darf die Original-Exception des Tests
+                # nicht maskieren (NIT aus den WI-P5-10-/WI-P5-11-Reviews).
+                # Singleton nicht im aktiven Zustand für Folgetests
+                # hinterlassen.
                 response = client.post("/execution/kill-switch/False")
-                assert response.status_code == 200
```

`tests/test_kill_switch.py` (NIT-A-Regression + NIT C):

- neue Klasse `TestIsolationFixtureTeardown` (2 Tests, bewusst
  Datei-Reihenfolge-abhängig — pytest führt Tests in Datei-Reihenfolge
  aus, Test 2 muss direkt nach Test 1 laufen):
  - `test_writes_log_entry_for_next_test` (ohne Marker):
    `routes.execution_log_store.add(decision_id="nit-a-1", run_id=
    "nit-a-run", symbol="BTCUSDT", side="buy", status="REJECTED",
    error="LIVE_EXECUTION_DISABLED")` → `count == 1`; der Write landet
    im tmp-gebundenen Store, nie in der echten `data/execution_log.json`
  - `test_opt_out_test_sees_clean_log_state`
    (`@pytest.mark.real_execution_log_state`): `count == 0` — ohne den
    neuen Teardown-Clear erbt der Test den In-Memory-Rest von Test 1
    (TDD-Red: `assert 1 == 0`)
- neue Klasse `TestCorruptedStateFileFallback` (1 Test, 3 nummerierte
  Assertion-Blöcke): korrumpiertes State-File (`{not valid json!!`) →
  (1) `KillSwitch(enabled=False, db_path=…)` ohne Exception,
  `is_active() is False` (Fallback auf Startzustand); (2) erneute
  Korruption + `KillSwitch(enabled=True, db_path=…)` →
  `is_active() is True` (Startzustand bleibt aktiv — pinnt die
  Fail-Open-Semantik); (3) `deactivate()` auf dieser Instanz repariert
  das File zu gültigem, parsebarem JSON mit `data["enabled"] is False`

## Verifikation

### TDD-RED (Tests vor dem conftest-Fix)

Test 2 rot mit `assert 1 == 0` (In-Memory-Rest von Test 1), Test 1
grün, Pin-Test C grün (pinnt bestehendes Verhalten und besteht daher
auch vor jeder Quelländerung):

```text
$ uv run pytest "tests/test_kill_switch.py::TestIsolationFixtureTeardown" "tests/test_kill_switch.py::TestCorruptedStateFileFallback" -v
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.4.2, pluggy-1.6.0 -- /root/smith/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /root/smith
configfile: pyproject.toml
plugins: asyncio-1.4.0, cov-6.3.0, anyio-4.14.2
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 3 items

tests/test_kill_switch.py::TestIsolationFixtureTeardown::test_writes_log_entry_for_next_test PASSED [ 33%]
tests/test_kill_switch.py::TestIsolationFixtureTeardown::test_opt_out_test_sees_clean_log_state FAILED [ 66%]
tests/test_kill_switch.py::TestCorruptedStateFileFallback::test_corrupted_state_file_falls_back_to_init_state PASSED [100%]

=================================== FAILURES ===================================
_____ TestIsolationFixtureTeardown.test_opt_out_test_sees_clean_log_state ______

self = <tests.test_kill_switch.TestIsolationFixtureTeardown object at 0x783aab1fcc20>

    @pytest.mark.real_execution_log_state
    def test_opt_out_test_sees_clean_log_state(self):
        """Test 2 (Marker-Opt-out) direkt nach Test 1 sieht einen leeren Store."""
        from trading_harness.api import routes

        # Ohne das WI-P5-14-Teardown-``clear()`` erbt dieser Test den
        # In-Memory-Eintrag von Test 1 (count == 1) → Test rot.
>       assert routes.execution_log_store.count == 0
E       AssertionError: assert 1 == 0
E        +  where 1 = <trading_harness.services.execution_store.ExecutionLogStore object at 0x783aab12f770>.count
E        +    where <trading_harness.services.execution_store.ExecutionLogStore object at 0x783aab12f770> = <module 'trading_harness.api.routes' from '/root/smith/src/trading_harness/api/routes.py'>.execution_log_store

tests/test_kill_switch.py:590: AssertionError
=========================== short test summary info ============================
FAILED tests/test_kill_switch.py::TestIsolationFixtureTeardown::test_opt_out_test_sees_clean_log_state
========================= 1 failed, 2 passed in 0.11s ==========================
```

### TDD-GREEN (nach dem conftest-Fix)

```text
$ uv run pytest "tests/test_kill_switch.py::TestIsolationFixtureTeardown" "tests/test_kill_switch.py::TestCorruptedStateFileFallback" -v
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.4.2, pluggy-1.6.0 -- /root/smith/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /root/smith
configfile: pyproject.toml
plugins: asyncio-1.4.0, cov-6.3.0, anyio-4.14.2
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 3 items

tests/test_kill_switch.py::TestIsolationFixtureTeardown::test_writes_log_entry_for_next_test PASSED [ 33%]
tests/test_kill_switch.py::TestIsolationFixtureTeardown::test_opt_out_test_sees_clean_log_state PASSED [ 66%]
tests/test_kill_switch.py::TestCorruptedStateFileFallback::test_corrupted_state_file_falls_back_to_init_state PASSED [100%]

============================== 3 passed in 0.03s ===============================
```

### `make check` (Endzustand)

```text
$ make check
uv run pytest -q
........................................................................ [  9%]
........................................................................ [ 18%]
........................................................................ [ 28%]
........................................................................ [ 37%]
........................................................................ [ 47%]
........................................................................ [ 56%]
........................................................................ [ 66%]
........................................................................ [ 75%]
........................................................................ [ 85%]
........................................................................ [ 94%]
........................................                                 [100%]
=============================== warnings summary ===============================
.venv/lib/python3.12/site-packages/fastapi/testclient.py:1
  /root/smith/.venv/lib/python3.12/site-packages/fastapi/testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
760 passed, 1 warning in 56.31s
uv run ruff check src tests
All checks passed!
uv run mypy src
Success: no issues found in 50 source files
MAKE_EXIT=0
```

760 = Baseline 757 + 3 neue Tests. Warning: bestehende
`StarletteDeprecationWarning` (unverändert).

### finally-Assert-AST-Scan (Scope-Proof NIT B)

Scan aller `tests/*.py` nach `assert`-Nodes innerhalb von
`finally`-Blöcken (`ast.Try.finalbody`), zuerst gegen den Pre-Fix-Baum
(HEAD), dann gegen den Worktree (Post-Fix):

```python
python3 - <<'EOF'
import ast, pathlib, subprocess

def finally_asserts(src: str) -> list[int]:
    hits = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Try):
            for stmt in node.finalbody:
                for child in ast.walk(stmt):
                    if isinstance(child, ast.Assert):
                        hits.append(child.lineno)
    return hits

print("== HEAD (pre-fix) tree: finally-Asserts in tests/ ==")
files = subprocess.run(
    ["git", "ls-tree", "-r", "--name-only", "HEAD", "tests/"],
    capture_output=True, text=True, check=True,
).stdout.split()
for f in sorted(files):
    if f.endswith(".py"):
        src = subprocess.run(
            ["git", "show", f"HEAD:{f}"], capture_output=True, text=True, check=True
        ).stdout
        for ln in finally_asserts(src):
            print(f"{f}:{ln}")

print("== current worktree (post-fix): finally-Asserts in tests/ ==")
found = False
for f in sorted(pathlib.Path("tests").rglob("*.py")):
    for ln in finally_asserts(f.read_text()):
        print(f"{f}:{ln}")
        found = True
if not found:
    print("(keine)")
EOF
```

Ergebnis — Pre-Fix (HEAD) genau 1 Treffer in der gesamten Suite (der
fixte Ort), Post-Fix 0:

```text
== HEAD (pre-fix) tree: finally-Asserts in tests/ ==
tests/test_api_security.py:354
== current worktree (post-fix): finally-Asserts in tests/ ==
(keine)
```

### `data/` vor und nach dem Gate (byte-identisch, keine `execution_log.json`)

```text
# VOR dem Gate (Workitem-Start)
$ ls -la data/
total 12
drwxr-xr-x  2 root root 4096 Aug 20 07:50 .
drwxr-xr-x 19 root root 4096 Aug 20 07:15 ..
-rw-r--r--  1 root root    0 Aug 13 13:22 .gitkeep
-rw-r--r--  1 root root  208 Aug 20 07:50 kill_switch.json
$ sha256sum data/kill_switch.json
1389df52f9a2e125a05a2ee96b13263870a234236506d2487c12cbc06d2383a9  data/kill_switch.json
$ cat data/kill_switch.json
{"enabled": false, "last_toggled_at": 1787212246.656831, "toggle_count": 2, "auto_trigger_enabled": true, "auto_trigger_threshold": 3, "anomaly_streak": 0, "auto_triggered": false, "trigger_reason": "manual"}

# NACH dem Gate (`make check`, 760 passed)
$ ls -la data/
total 12
drwxr-xr-x  2 root root 4096 Aug 20 07:50 .
drwxr-xr-x 19 root root 4096 Aug 20 07:15 ..
-rw-r--r--  1 root root    0 Aug 13 13:22 .gitkeep
-rw-r--r--  1 root root  208 Aug 20 07:50 kill_switch.json
$ sha256sum data/kill_switch.json
1389df52f9a2e125a05a2ee96b13263870a234236506d2487c12cbc06d2383a9  data/kill_switch.json
$ ls data/execution_log.json
ls: cannot access 'data/execution_log.json': No such file or directory
```

`kill_switch.json` byte-identisch vor/nach (gleicher sha256, gleiche
mtime `Aug 20 07:50`, `enabled: false`, `toggle_count: 2`); keine
`data/execution_log.json` erzeugt — die Test-Suite (inkl. der
Teardown-Clears, die nur auf tmp-Pfaden wirken) berührt die echten
State-Pfade nicht.

## Abweichungen (im Workitem-Scope deklariert)

- `tests/test_api_security.py` — nicht in der Workitem-Dateiliste,
  aber NIT B lebt dort (einziger finally-Assert der Suite, per
  AST-Scan nachgewiesen, siehe oben).
- `README.md` — Test-Zähler Abschnitt 8: 757 → 760 (gleiche
  Deklarationsabweichung wie WI-P5-15; Doku-Aktualität desselben
  Abschnitts, keine semantische Änderung).
- Sonstige Abweichungen: keine. `src/` (u. a. `kill_switch.py`,
  `execution_store.py`, `routes.py`), `pyproject.toml`, `Makefile`,
  `docker-compose.yml` und `tests/test_api_execution.py` unberührt; die
  bekannten kosmetischen NITs (fehlende End-zeilenumbrüche, dir-fsync-
  Hardening, CWD-relative Default-Pfade) bleiben bewusst
  Dokumentierungs-Kandidaten. Keine Sicherheitsgrenzen-/
   Verhaltensänderung: der Fail-Open-Fallback wird GEpinnt, nicht
   geändert, Live-Execution bleibt deaktiviert, Kill-Switch-Semantik
   unverändert.

## Commit

Conventional Commit (einziger Commit dieses Workitems, explizit
gestagte Pfade nur):

```
test(execution): NIT-Bundle P5-10/P5-11 (conftest-Teardown,
finally-Cleanup, Pin-Test korrumpiertes State-File) (WI-P5-14)
```

`git diff --cached --stat` im Stage-Zustand bei Commit-Erzeugung
(byte-identisch wiedergegeben; die Zeilenangabe der Evidenzdatei ist
deren Finallänge):

```
 README.md                                     |   2 +-
 docs/handoff.md                               | 108 +++++++--
 evidence/wi-p5-14-nit-bundle-test-evidence.md | 323 ++++++++++++++++++++++++++
 tests/conftest.py                             |  10 +
 tests/test_api_security.py                    |   7 +-
 tests/test_kill_switch.py                     |  90 ++++++-
 6 files changed, 523 insertions(+), 17 deletions(-)
```

Da ein Commit seinen eigenen SHA nicht enthalten kann, ist der SHA
über `git log --grep="WI-P5-14" --format='%H'` bzw. `git log -1
--format='%H'` (am Commit, der diese Datei enthält) ermittelbar.

Worktree nach dem Commit: `git status -s` zeigt ausschließlich die
vorbestehenden `.omo/`-Laufzeitdateien (ungetrackt/geändert); keine
andere Datei verlässt oder erreicht den Index, `data/` bleibt
unverändert (siehe Daten-Prüfung oben).
