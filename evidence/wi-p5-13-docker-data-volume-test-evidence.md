# WI-P5-13 — Docker `./data`-Bind-Mount für api-State-Persistenz (Test Evidence)

Datum: 2026-08-20 · Review-Bezug: Review-MINOR-3 von WI-P5-10

## Ziel

`docker-compose.yml` mountete für den `api`-Service nur `./config`, `./prompts`
und `./schemas` — alle read-only. Der Kill-Switch-State wird vom API-Prozess
nach `data/kill_switch.json` geschrieben (`Settings.kill_switch_state_path`,
Default relativ zum WORKDIR `/app`, also `/app/data/kill_switch.json` im
Container), und ab WI-P5-15 zusätzlich das Execution-Audit-Log nach
`data/execution_log.json`. Ohne schreibbaren Mount lag dieser State im
Container-Layer und ging bei jeder Container-Recreation verloren
(Review-MINOR-3 von WI-P5-10, Finding (c) des WI-P5-10-Reviews).

Fix: genau ein schreibbarer Bind-Mount `./data:/app/data` für den `api`-
Service. Config + Docs + Evidence-Change, kein Python-Code- oder Test-Change.

## Änderung

`docker-compose.yml`, `volumes`-Liste des `api`-Services — eine Zeile, nach
dem `./schemas`-Eintrag, ohne `:ro`:

```diff
     volumes:
       - ./config:/app/config:ro
       - ./prompts:/app/prompts:ro
       - ./schemas:/app/schemas:ro
+      - ./data:/app/data
```

Die drei bestehenden read-only-Mounts, Ports (`8080:8080`), `depends_on`,
`env_file` sowie die postgres/redis-Services und die top-level `volumes`
bleiben unverändert. Der Host-Ordner `./data` existiert und enthält nur
`.gitkeep`; `.gitignore` schließt `data/*` aus, ausgenommen `data/.gitkeep`
(State-Dateien werden nie committet).

Dokumentiert in README.md (Abschnitt 5 "Schnellstart", neuer Absatz
"Zustands-Persistenz") und `docs/handoff.md` (Punkt 8).

## Umgebung

```text
docker compose version → Docker Compose version v5.4.0
docker info           → Server OK: 29.7.1 (Daemon läuft)
```

## Verifikation

### `docker compose config --quiet`

```text
$ docker compose config --quiet
(keine Ausgabe)
EXIT=0
```

### `docker compose build`

```text
$ docker compose build
...
#17 writing image sha256:d14826ec009ec4a27cdeacc7bba9d89e38c73637ca2d5cd0c2b53c1789dde55d done
#17 naming to docker.io/library/smith-api done
 Image smith-api Built
BUILD_EXIT=0
```

### Smoke-Test (Persistence nach Container-Recreation)

Vorherige Zustand: `data/` enthält nur `.gitkeep`; `TRADE_API_KEY`/
`READ_API_KEY` sind in `.env` nicht gesetzt → `/execution/kill-switch/{enabled}`
erfordert nach `api/security.py` keinen Key.

```text
$ docker compose up -d
Network smith_default Created
Volume smith_redis-data Created / Volume smith_postgres-data Created
Container smith-postgres-1 Healthy
Container smith-redis-1 Healthy
Container smith-api-1 Started
UP_EXIT=0
```

Health-Poll (4-s-Intervall):

```text
HEALTH_OK after ~4s: {"status":"ok","live_execution_enabled":false,"kill_switch":true}
```

**Schritt 1 — Kill Switch ON, Host-Datei entsteht:**

```text
$ curl -s -X POST http://localhost:8080/execution/kill-switch/true
{"kill_switch":true}
$ curl -s http://localhost:8080/execution/status
{"live_execution_enabled":false,"kill_switch":true,"execution_logs_count":0}
$ cat data/kill_switch.json
{"enabled": true, "last_toggled_at": 1787212233.2742894, "toggle_count": 1, "auto_trigger_enabled": true, "auto_trigger_threshold": 3, "anomaly_streak": 0, "auto_triggered": false, "trigger_reason": "manual"}
```

Der Container schrieb die State-Datei in den Bind-Mount → sichtbar auf dem
Host.

**Schritt 2 — Container-Recreation (Kern-Proof):**

```text
$ docker compose up -d --force-recreate api
Container smith-api-1 Recreate
Container smith-api-1 Recreated
Container smith-api-1 Started
RECREATE_EXIT=0
```

Health-Poll: `HEALTH_OK after ~4s: {"status":"ok","live_execution_enabled":false,"kill_switch":true}`

```text
$ curl -s http://localhost:8080/execution/status   # frischer Container liest persistierten State
{"live_execution_enabled":false,"kill_switch":true,"execution_logs_count":0}
$ cat data/kill_switch.json                        # Host-Datei nach Recreation
{"enabled": true, "last_toggled_at": 1787212233.2742894, "toggle_count": 1, "auto_trigger_enabled": true, "auto_trigger_threshold": 3, "anomaly_streak": 0, "auto_triggered": false, "trigger_reason": "manual"}
```

Host-Datei byte-identisch zum Vorher-Zustand (gleiche `last_toggled_at`,
gleiche `toggle_count: 1`) — und der neu erzeugte Container startet mit
`kill_switch: true`, weil er den State aus `data/kill_switch.json` lädt.
Persistenz überlebt die Container-Recreation.

**Schritt 3 — Kill Switch OFF:**

```text
$ curl -s -X POST http://localhost:8080/execution/kill-switch/false
{"kill_switch":false}
$ curl -s http://localhost:8080/execution/status
{"live_execution_enabled":false,"kill_switch":false,"execution_logs_count":0}
$ cat data/kill_switch.json
{"enabled": false, "last_toggled_at": 1787212246.656831, "toggle_count": 2, "auto_trigger_enabled": true, "auto_trigger_threshold": 3, "anomaly_streak": 0, "auto_triggered": false, "trigger_reason": "manual"}
```

**Schritt 4 — Stack stoppen (ohne `-v`):**

```text
$ docker compose down
Container smith-api-1 Stopped / Removed
Container smith-postgres-1 Stopped / Removed
Container smith-redis-1 Stopped / Removed
Network smith_default Removed
DOWN_EXIT=0
$ docker ps --format '{{.Names}}' | grep -c smith
0
```

Kein smith-Stack läuft mehr; der Kill Switch ist deaktiviert
(`"enabled": false`). Named Volumes `smith_postgres-data` /
`smith_redis-data` bleiben erhalten (bewusst, `down` ohne `-v`).
Restzustand: `data/kill_switch.json` verbleibt auf dem Host im
deaktivierten State (gitignoriert, inaktiv).

### `make check`

```text
=============================== warnings summary ===============================
.venv/lib/python3.12/site-packages/fastapi/testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
754 passed, 1 warning in 52.47s
uv run ruff check src tests
All checks passed!
uv run mypy src
Success: no issues found in 50 source files
CHECK_EXIT=0
```

Warning: bestehende `StarletteDeprecationWarning`
(`fastapi/testclient.py`, seit WI-P5-11 unverändert).

## Abweichungen

- **Host-Ports 5432/6379 belegt:** Die Port-Präferenz (8080/5432/6379 frei)
  war buchstäblich nicht erfüllt — 5432/6379 werden von einem fremden,
  nicht zu diesem Repository gehörenden Docker-Stack (`aurora`, seit
  6–8 Tagen aktiv) belegt. Es gab dennoch **keinen Port-Konflikt** für den
  smith-Stack: `docker-compose.yml` publiziert nur Port 8080 (api);
  postgres/redis haben kein Host-Port-Mapping und werden über die interne
  Compose-Netzwerk-DNS (`postgres`/`redis` aus `DATABASE_URL` in `.env`)
  erreicht. Der Smoke-Test war daher vollständig durchführbar und wurde
  ohne Abstriche ausgeführt.
- Keine weiteren Abweichungen: `docker compose config --quiet`,
  `docker compose build`, Smoke-Test und `make check` liefen vollständig.

## Commit

Diese Datei ist Teil des WI-P5-13-Commits (Conventional Commit
`feat(docker): … (WI-P5-13)`). Da ein Commit seinen eigenen SHA nicht
enthalten kann, ist der SHA über `git log --grep="WI-P5-13" --format='%H'`
bzw. `git log -1 --format='%H'` (am Commit, der diese Datei enthält)
ermittelbar.
