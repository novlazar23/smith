# WI-P5-12 — Kill-Switch-State-Write via `tempfile.mkstemp` (Test Evidence)

Datum: 2026-08-20 · Review-Bezug: Review-MINOR-2 von WI-P5-10

## Ziel

`KillSwitch._save_state` verwendete einen deterministischen Tmp-Namen
(`kill_switch.json.tmp`) für alle Writes. Zwei `KillSwitch`-Instanzen, die
sich einen State-Pfad teilen (z. B. API- und CLI-Prozess, zwei
Singletons in einem Prozess), truncate-/überschreiben sich gegenseitig die
Tmp-Datei → Lost Updates, gemischte Dokumente, `FileNotFoundError` beim
`os.replace` (stillschweigend verschluckt durch `except OSError`) und
State-Divergenz zwischen In-Memory-Zustand und Datei.

Fix: eindeutige Tmp-Datei pro Writer via
`tempfile.mkstemp(dir=<State-Verzeichnis>, prefix=<Name>.", suffix=".tmp")`
— selbes Verzeichnis → atomares `os.replace` bleibt erhalten.

## TDD-Red

Zuerst hinzugefügter Test (rot gegen alten Code):
`tests/test_kill_switch.py::TestKillSwitchMultiWriterIsolation::test_multi_writer_collision_no_lost_update`

Erzwungenes Interleaving (events, kein Timing-Lottery): Writer A
(`activate()`) schreibt doc_a in die Tmp-Datei und blockiert in
`os.fsync` (erster Aufruf); Writer B (`deactivate()`) trunciert die
geteilte Tmp-Datei (alter Code), schreibt doc_b; B's `os.replace`
(erster Aufruf) gibt A frei und wartet auf A's Replace; A's Replace
(zweiter Aufruf) konsumiert die geteilte Tmp-Datei, die zu diesem
Zeitpunkt doc_b enthält. Ergebnis: Datei enthält B's Dokument, obwohl A
der zuletzt erfolgreiche Replacer war → Lost Update.

Auszug (roter Lauf gegen alten Code):

```text
E       AssertionError: assert {'anomaly_str...': False, ...} == {'anomaly_str...': False, ...}
E         Omitting 5 identical items, use -vv to show
E         Differing items:
E         {'trigger_reason': None} != {'trigger_reason': 'manual'}
E         {'enabled': False} != {'enabled': True}
E         {'last_toggled_at': 1787209829.7647269} != {'last_toggled_at': 1787209829.7644026}
FAILED tests/test_kill_switch.py::TestKillSwitchMultiWriterIsolation::test_multi_writer_collision_no_lost_update
========================= 1 failed, 28 passed in 0.32s =========================
```

Stabilität des Reds: 5/5 Einzel-Läufe des Kollisionstests gegen alten
Code fehlgeschlagen (`1 failed in 0.06–0.07s`) — deterministisch.

## Fix

`src/trading_harness/services/kill_switch.py::_save_state`:

- `tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")`
  → eindeutige Tmp-Datei pro Write im selben Verzeichnis (atomares
  `os.replace` bleibt erhalten)
- `os.fdopen(tmp_fd, "w")` + `json.dump` + `f.flush()` + `os.fsync(f.fileno())`
  (Schreibpfad wie bisher, fd-basiert)
- Modus-Erhaltung: `mkstemp` legt 0600 an → explizites `os.chmod` auf
  `path.stat().st_mode & 0o777` (Ziel existiert) bzw. 0644 (Neuanlage)
- `OSError`: fehlgeschlagene Tmp-Datei wird best-effort entfernt
  (`os.unlink` in eigenem `try/except OSError: pass`); geöffneter FD wird
  in `finally` geschlossen (kein FD-Leak)
- Öffentliche API, JSON-Format und `except OSError: pass`-Semantik
  (Persistenzfehler nicht kritisch) unverändert

## TDD-Green

```text
tests/test_kill_switch.py::TestKillSwitchMultiWriterIsolation::test_multi_writer_collision_no_lost_update PASSED [ 93%]
tests/test_kill_switch.py::TestKillSwitchMultiWriterIsolation::test_concurrent_writers_stress PASSED [ 96%]
tests/test_kill_switch.py::TestKillSwitchMultiWriterIsolation::test_state_file_mode_0644_new_and_overwrite PASSED [100%]
============================== 29 passed in 0.27s ==============================
```

10× Stabilität (alle drei neuen Tests, 10 aufeinanderfolgende Läufe):
jeder Lauf `3 passed in 0.17–0.19s`, 0 Fehlschläge.

Neue Tests (alle in `tests/test_kill_switch.py`, Klasse
`TestKillSwitchMultiWriterIsolation`):

1. `test_multi_writer_collision_no_lost_update` — deterministischer
   Multi-Writer-Kollisionstest (erzwungenes Interleaving); Invariante:
   die Datei muss exakt das vollständige Snapshot des zuletzt
   erfolgreich persistierenden Writers enthalten
2. `test_concurrent_writers_stress` — 2 Instanzen, 4 Threads × 25 Toggles
   auf einem State-Pfad; Invarianten: valides JSON mit vollständigem
   Key-Set, `toggle_count == 50` (kein Lost Update), keine `*.tmp*`-
   Rückstände, Reload-Konsistenz einer frischen Instanz
3. `test_state_file_mode_0644_new_and_overwrite` — 0644-Modus bei
   Neuanlage UND Überschreiben (guard gegen `mkstemp`-Default 0600)

## make check

```text
uv run pytest -q            → 754 passed, 1 warning in 52.48s
uv run ruff check src tests → All checks passed!
uv run mypy src             → Success: no issues found in 50 source files
```

Warning: bestehende `StarletteDeprecationWarning`
(`fastapi/testclient.py`, seit WI-P5-11 unverändert).

## Verbleibende Punkte

- Docker mountet `./data` nicht (WI-P5-10 Finding (c)) — Persistenz
  überlebt keine Container-Recreation; bleibt separates Workitem
  (Container-Änderung, erfordert Freigabe)
- `ExecutionLogStore()` in `routes.py` hat weiterhin kein `db_path`
  (Audit-Log-Persistenz, WI-P5-10, separates Workitem)
- `kill_switch_default` (Settings) bleibt unverdrahtet (Verhaltensänderung
  der Sicherheitsgrenze, nur mit expliziter Freigabe)

## Commit

Diese Datei ist Teil des WI-P5-12-Commits (Conventional Commit
`fix(execution): … (WI-P5-12)`). Da ein Commit seinen eigenen SHA nicht
enthalten kann, ist der SHA über `git log --grep="WI-P5-12" --format='%H'`
bzw. `git log -1 --format='%H'` (am Commit, der diese Datei enthält)
ermittelbar.
