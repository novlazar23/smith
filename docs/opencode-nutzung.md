# OpenCode-Nutzungsanleitung

Diese Anleitung beschreibt, wie das Projekt mit OpenCode eigenständig eingerichtet, entwickelt,
geprüft und zwischen Geräten übergeben wird. OpenCode benötigt keine Codex-, Harness- oder
Conversation-Memory-Daten. Der vollständige gemeinsame Entwicklungsstand liegt in Git.

## 1. Voraussetzungen

- Git mit Zugriff auf das private Repository `novlazar23/smith`
- OpenCode
- Für lokale Python-Entwicklung: `uv` 0.11.x
- Alternativ für den Containerbetrieb: Docker mit Docker Compose v2
- Zugang zu einem von OpenCode unterstützten Modellprovider

Zugangsdaten werden ausschließlich auf dem jeweiligen Gerät eingerichtet. Sie gehören weder in
Git noch in `.env.example`.

## 2. Ersteinrichtung

```bash
git clone https://github.com/novlazar23/smith.git
cd smith
./scripts/bootstrap.sh
make check
```

Der Bootstrap erzeugt bei Bedarf eine lokale `.env`, installiert die in `uv.lock` gesperrten
Abhängigkeiten und verwendet die in `.python-version` festgelegte Python-Version. `.env` und
`.venv` werden nicht committed.

Für einen reinen Docker-Start:

```bash
./scripts/bootstrap.sh --docker
```

Die API ist anschließend unter `http://localhost:8080` erreichbar. Der Healthcheck lautet:

```bash
curl http://localhost:8080/health
```

## 3. OpenCode einrichten und starten

Die persönliche Provider- und Modellanmeldung erfolgt über OpenCode:

```bash
opencode providers
```

Danach OpenCode immer im Repository-Root starten:

```bash
opencode
```

Die Modellwahl wird absichtlich nicht im Repository festgeschrieben. Dadurch kann jedes Gerät
einen eigenen unterstützten Provider verwenden, ohne Projektdateien oder Secrets zu verändern.

Beim Start lädt OpenCode automatisch:

- `AGENTS.md` – verbindliche Architektur-, Entwicklungs- und Sicherheitsregeln
- `opencode.json` – Projektberechtigungen und Kontextkomprimierung
- `.opencode/commands/` – wiederverwendbare Projektbefehle
- `docs/handoff.md` – aktueller, in Git gespeicherter Übergabestand

## 4. Entwicklung fortsetzen

Nach dem Start in OpenCode:

```text
/resume
```

`/resume` liest Git-Status, letzte Commits, offene Änderungen, `AGENTS.md`, README und den Handoff.
Wenn ein aktiver Auftrag dokumentiert ist, wird er fortgesetzt. Andernfalls wählt OpenCode den
höchst priorisierten sicheren, noch offenen Punkt aus der Entwicklungsreihenfolge im README.

Eine konkrete Aufgabe kann direkt mitgegeben werden:

```text
/resume Implementiere API-Authentifizierung für alle mutierenden Endpunkte.
```

OpenCode soll Änderungen vollständig umsetzen, passende Tests hinzufügen, die Dokumentation
aktualisieren und den Qualitäts-Gate ausführen. Live-Execution bleibt außerhalb des zulässigen
Umfangs, solange sie nicht ausdrücklich freigegeben und technisch abgesichert wurde.

## 5. Prüfen und reparieren

In OpenCode:

```text
/check
```

Oder im Terminal:

```bash
./scripts/bootstrap.sh --check
```

Der Gate umfasst:

- vollständige Pytest-Suite
- Ruff-Linting
- Mypy-Typprüfung

Bei Containeränderungen zusätzlich:

```bash
docker compose config --quiet
docker build -t smith:local .
```

OpenCode darf Prüfungen nicht abschwächen oder überspringen, um einen grünen Stand zu erzeugen.

## 6. Übergabe an ein anderes Gerät

In OpenCode:

```text
/handoff
```

Der Befehl prüft den vollständigen Stand, aktualisiert `docs/handoff.md`, kontrolliert den Diff auf
lokale Daten und erstellt einen Conventional Commit. Ein Push benötigt weiterhin eine ausdrückliche
Bestätigung.

Danach veröffentlichen:

```bash
git push
```

Auf dem anderen Gerät:

```bash
git clone https://github.com/novlazar23/smith.git
cd smith
./scripts/bootstrap.sh
opencode
```

Anschließend wieder `/resume` ausführen. Bei einem bestehenden Checkout vorher synchronisieren:

```bash
git status -sb
git pull --ff-only
./scripts/bootstrap.sh
opencode
```

## 7. Sicherheitsgrenzen

OpenCode darf innerhalb des Repositorys autonom lesen, editieren, testen und recherchieren. Die
versionierte Konfiguration erzwingt zusätzliche Grenzen:

- `.env` darf nicht gelesen oder editiert werden; `.env.example` bleibt als Vorlage zugänglich.
- Zugriffe außerhalb des Worktrees sind blockiert.
- `git push` benötigt eine Bestätigung.
- Force-Push und `git reset --hard` sind blockiert.
- rekursives Löschen und das Entfernen persistenter Docker-Volumes benötigen eine Bestätigung.
- Live-Execution, Kill-Switch und deterministische Risk Limits dürfen nicht ohne ausdrückliche
  Freigabe verändert oder abgeschwächt werden.

Niemals committen:

- API- und Exchange-Schlüssel
- Passwörter oder private Zertifikate
- `.env`
- `.venv` und Tool-Caches
- Datenbank-Dumps oder Docker-Volumes

## 8. Was Git überträgt

Git überträgt Quellcode, Tests, Dokumentation, Konfigurationen, Prompts, Schemas, Dependency-Lock,
Projektregeln und `docs/handoff.md`. Nicht übertragen werden Provider-Anmeldung,
OpenCode-Sitzungsverlauf, lokale Datenbanken, Docker-Volumes und Secrets.

Wenn derselbe Datenbestand auf mehreren Geräten benötigt wird, ist dafür ein separater,
verschlüsselter Backup-/Restore-Prozess erforderlich.

## 9. Fehlerbehebung

### `uv` fehlt

`uv` gemäß der offiziellen Dokumentation installieren und danach erneut ausführen:

```bash
./scripts/bootstrap.sh
```

### Abhängigkeiten oder Python-Version weichen ab

```bash
uv sync --frozen --all-extras
make check
```

`--frozen` verhindert, dass das Lockfile während der Installation still verändert wird.

### OpenCode erkennt die Projektbefehle nicht

OpenCode im Repository-Root starten und die aufgelöste Konfiguration prüfen:

```bash
opencode debug config
```

Erwartete Befehle sind `check`, `handoff` und `resume`.

### Docker-Build kann Pakete nicht auflösen

Zuerst Netzwerk- und DNS-Zugriff des Docker-Daemons prüfen. Ein erfolgreicher lokaler `uv sync`
belegt nicht automatisch, dass auch das isolierte Docker-Build-Netz Zugriff auf PyPI besitzt.

### Arbeitsstand ist unklar

```bash
git status -sb
git log --oneline --decorate -10
```

Danach `docs/handoff.md` lesen und in OpenCode `/resume` ausführen.
