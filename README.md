# Evolutionary Trading Harness

Ein modularer, auditierbarer Trading-Research-Service mit Multi-Agenten-Analyse, deterministischer Risk Engine und kontrollierter Agenten-Evolution.

> **Status:** Entwicklungsgrundlage / Shadow-Trading-MVP. Live-Execution ist standardmäßig deaktiviert.

---

## 1. Ziel

Dieses Repository soll zu einer dauerhaft laufenden Trading- und Agentenplattform ausgebaut werden.

Das System trennt strikt:

1. Marktdaten
2. Snapshot-Erzeugung
3. Agenten-Analyse
4. adversariales Review
5. Consensus
6. deterministische Risikoprüfung
7. Trading-Entscheidung
8. Execution
9. Performance-Messung
10. Agenten-Evolution

Codex dient als Entwicklungs- und Meta-Orchestrator. Die produktive Runtime läuft unabhängig von Codex.

---

# 2. Entwicklungsanweisungen für Codex

## Rolle

Du arbeitest als leitender Softwarearchitekt und Entwickler dieses Repositories.

Dein Ziel ist nicht, möglichst schnell Trading-Funktionalität zu erzeugen, sondern ein reproduzierbares, testbares, sicher begrenztes und evolutionär erweiterbares Trading-Research-System aufzubauen.

Arbeite selbstständig weiter, bis die jeweilige Aufgabe vollständig implementiert, getestet, dokumentiert und überprüft ist.

Wenn Anforderungen nicht bis ins Detail definiert sind:

- triff konservative, nachvollziehbare Architekturentscheidungen,
- dokumentiere Annahmen,
- bevorzuge Sicherheit, Reproduzierbarkeit und Testbarkeit,
- vermeide unnötige Abhängigkeiten,
- ändere keine Sicherheitsgrenzen stillschweigend.

## Nicht verhandelbare Architekturregeln

### Trennung der Verantwortlichkeiten

Folgende Bereiche müssen logisch und technisch getrennt bleiben:

- GENERATION
- EVALUATION
- PROMOTION
- TRADING
- EXECUTION

Kein einzelner Agent oder LLM-Aufruf darf einen neuen Agenten erzeugen, selbst bewerten, selbst promoten und anschließend Live-Trades ausführen.

### Deterministische Sicherheitsgrenzen

Folgende Regeln werden niemals einem LLM überlassen:

- maximales Risiko pro Trade
- maximales Tagesverlustlimit
- maximales Portfoliorisiko
- maximaler Hebel
- maximale Positionsanzahl
- Kill Switch
- Order-Deduplizierung
- erlaubte Symbole
- erlaubte Exchanges
- minimale Risk/Reward-Anforderung
- maximal tolerierte Slippage

LLMs dürfen diese Werte lesen und berücksichtigen, aber nicht überschreiben.

### Execution

Der Execution Service:

- analysiert keine Märkte,
- erzeugt keine neuen Trades,
- verändert keine Positionsgrößen,
- verändert keine Richtung,
- verändert keine Stops,
- verändert keine Risk Limits.

Live-Execution muss standardmäßig deaktiviert bleiben.

### Agenten

Neue Agenten beginnen als `GENERATED`.

Lifecycle:

```text
GENERATED
  -> CANDIDATE
  -> CHALLENGER
  -> ACTIVE
  -> CHAMPION

ACTIVE / CHAMPION
  -> PROBATION
  -> RETIRED

Fehlgeschlagene Kandidaten
  -> REJECTED
```

Agenten konkurrieren ausschließlich innerhalb derselben Kategorie.

Beispiel:

- Technical gegen Technical
- Elliott gegen Elliott
- Orderflow gegen Orderflow

Ein Elliott-Agent ersetzt niemals direkt einen Technical-Agenten.

## Agenten-Evolution

Neue Agenten dürfen entstehen durch:

- Mutation
- Recombination
- Specialization
- Simplification
- Diversity Injection

Bevorzugt werden kleine, messbare Änderungen von 1-3 Eigenschaften pro Generation.

Jeder neue Agent benötigt:

- eindeutige ID
- Generation
- Kategorie
- Parent-IDs
- Hypothese
- Mutationstyp
- Prompt-Version
- Genome
- erwarteten Vorteil
- erwartete Failure Modes

Prompts dürfen nicht nur kosmetisch umformuliert werden. Eine neue Variante muss eine funktional überprüfbare Hypothese darstellen.

## Bewertung

Kein Agent wird aufgrund eines einzelnen Trades bewertet.

Bewertungsdimensionen:

### Prediction Quality

- Brier Score
- Log Loss
- Directional Accuracy
- Calibration Error
- Precision / Recall, wenn sinnvoll

### Trading Utility

- Expectancy
- Profit Factor
- Sharpe
- Sortino
- Maximum Drawdown
- MFE
- MAE

### Robustness

- Regime Robustness
- Cross-Asset Robustness
- Out-of-Sample Performance
- Walk-Forward Stability

### Operational Quality

- Hallucination Rate
- Missing-Data Handling
- Schema Compliance
- Determinismus / Reproduzierbarkeit
- Laufzeit
- Token-/Compute-Kosten

### Ensemble Contribution

Ein individuell mittelmäßiger Agent kann wertvoll bleiben, wenn seine Fehler nicht mit den dominierenden Agenten korreliert sind.

Bewerte deshalb zusätzlich:

- Signal Correlation
- Error Correlation
- Marginal Ensemble Contribution
- Diversity Contribution

## Champion / Challenger

Ein Challenger darf einen Incumbent derselben Kategorie nur ersetzen, wenn:

- Mindeststichprobe erreicht
- Out-of-Sample verbessert
- Walk-Forward verbessert
- Shadow Mode bestanden
- Ensemble-Beitrag nicht verschlechtert
- Security Checks bestanden
- Verbesserung statistisch plausibel
- Promotion Margin erreicht

Kein vollständiger Austausch einer Kategorie in einer Generation.

Standard:

```yaml
max_replacement_per_generation: 0.20
```

## Regime-Spezialisierung

Nicht ausschließlich nach einem globalen Mittelwert selektieren.

Behalte geeignete Spezialisten für:

- strong_bull
- weak_bull
- range
- weak_bear
- strong_bear
- high_volatility
- low_volatility
- crash
- recovery

Ein Agent mit etwas schwächerem Gesamtscore darf bestehen bleiben, wenn er in einem wichtigen Regime einen hohen marginalen Nutzen für das Ensemble besitzt.

## Historische Evaluation

Tests müssen zeitlich korrekt sein.

Vermeide:

- Look-Ahead Bias
- Leakage
- Survivorship Bias
- Data Snooping
- nachträgliche Benchmark-Anpassung

Verwende:

1. Historical Evaluation
2. Out-of-Sample
3. Walk-Forward
4. Shadow Trading
5. Champion-vs-Challenger

Erst danach ist eine Promotion zulässig.

## Performance-Metriken

Metriken werden durch deterministischen Code berechnet.

LLMs dürfen:

- Ergebnisse interpretieren,
- Failure Modes erklären,
- Verbesserungshypothesen erzeugen.

LLMs dürfen nicht:

- ihre eigenen Scores frei festlegen,
- Bewertungsmetriken nachträglich verändern,
- fehlende Outcomes erfinden.

## Agent Genome

Mindestens:

```yaml
agent_genome:
  id:
  generation:
  parent_agents: []
  category:
  status:
  prompt_version:
  reasoning_style:
  indicators: []
  timeframes: []
  feature_preferences: []
  statistical_methods: []
  weighting_strategy:
  confidence_calibration:
  risk_attitude:
  context_window_strategy:
  output_schema:
  model_profile:
  temperature:
  created_at:
```

## LLM Routing

Die Runtime soll gegen einen OpenAI-kompatiblen Gateway arbeiten.

Bevorzugte logische Modellprofile:

```text
local-fast
local-main
local-critic
```

Die konkrete Provider-/Modellzuordnung gehört in Konfiguration und nicht in Agent-Prompts.

Implementiere den LLM-Client so, dass die Runtime austauschbar bleibt.

## Audit Trail

Jede wichtige Aktion benötigt einen Audit-Eintrag:

- Agent erzeugt
- Agent mutiert
- Evaluation gestartet
- Evaluation abgeschlossen
- Promotion
- Retirement
- Risk Reject
- Decision
- Execution Attempt
- Kill Switch

Speichere mindestens:

- Timestamp
- Actor
- Action
- Entity ID
- Input Hash
- Config Version
- Ergebnis
- Reason

## Reproduzierbarkeit

Jeder Trading Run muss auf einen unveränderlichen Snapshot zeigen.

Alle Agenten eines Runs analysieren denselben `snapshot_id`.

Speichere:

- Snapshot Hash
- Agent ID
- Agent Generation
- Prompt Hash
- Modellprofil
- Modellparameter
- Resultat
- Timestamp

## Datenbank

Ziel:

PostgreSQL + TimescaleDB.

Geplante Tabellen:

```text
market_snapshots
agents
agent_versions
agent_genomes
agent_runs
agent_predictions
market_outcomes
agent_scores
evaluation_runs
evaluation_results
populations
population_members
promotions
retirements
trading_decisions
orders
executions
positions
risk_events
audit_log
```

Migrationen müssen versioniert sein.

## Eventing

Für den MVP genügt Redis.

Die Architektur soll später eine Migration zu Redis Streams oder NATS erlauben.

Keine harte Business-Logik direkt in Queue-spezifische Handler einbauen.

## API

Zielendpunkte:

```text
GET  /health
GET  /agents
GET  /agents/{id}
POST /agents/generate

GET  /populations
GET  /populations/{category}

POST /snapshots
GET  /snapshots/{id}

POST /runs
GET  /runs/{id}

POST /evaluations
GET  /evaluations/{id}

POST /evolution/run

POST /risk/evaluate

GET  /decisions
GET  /decisions/{id}

POST /execution/orders
POST /kill-switch
```

## Tests

Jede Änderung muss passende Tests erhalten.

Mindestens:

- Unit Tests
- Risk Engine Tests
- Promotion Policy Tests
- Schema Tests
- API Tests
- Regression Tests für bekannte Fehler

Besonders sicherheitskritische Bereiche benötigen negative Tests.

Beispiele:

- Risk Limit überschritten -> REJECT
- Hebel überschritten -> REJECT
- Kill Switch aktiv -> REJECT
- unbekanntes Symbol -> REJECT
- doppelte decision_id -> REJECT
- Challenger ohne OOS -> keine Promotion
- Challenger ohne Shadow Pass -> keine Promotion
- falsche Kategorie -> kein Replacement

## Coding Standards

- Python 3.12+
- Typannotationen
- Pydantic für API-/Domain-Schemas
- FastAPI für HTTP
- kleine, testbare Funktionen
- klare Domain-Objekte
- keine Business-Logik in HTTP-Routen
- keine Secrets im Repository
- strukturierte Logs
- UTC für persistierte Zeitstempel
- IDs müssen global eindeutig sein

## Sicherheitsregeln für Secrets

Nie committen:

- Exchange API Keys
- Exchange Secrets
- LLM API Keys
- Datenbankpasswörter
- private Zertifikate

Nur `.env`, Secret Store oder Container Secrets.

`.env` ist in `.gitignore`.

## Arbeitsweise für Codex

Bei jeder größeren Aufgabe:

1. Bestand analysieren
2. Ziel und betroffene Module bestimmen
3. Implementierung durchführen
4. Tests ergänzen
5. Tests ausführen
6. Fehler beheben
7. Dokumentation aktualisieren
8. Security-/Regression-Review durchführen
9. Änderung zusammenfassen

Keine halbfertigen Platzhalter als „fertig“ deklarieren.

TODOs sind nur zulässig, wenn:
- sie nicht sicherheitskritisch sind,
- sie klar dokumentiert werden,
- die aktuelle Implementierung trotzdem konsistent funktioniert.

---

# 3. Architektur

```text
                         Codex
                           |
                       Git / API
                           |
                           v
+-----------------------------------------------------------+
|                   TRADING HARNESS                         |
|                                                           |
| Market Data -> Snapshot -> Agent Runtime                  |
|                               |                           |
|       +-----------------------+--------------------+      |
|       |                       |                    |      |
|   Technical              Macro/News            Orderflow  |
|       |                       |                    |      |
|       +-----------------------+--------------------+      |
|                               v                           |
|                       Trading Orchestrator                |
|                               |                           |
|                      Bull / Bear / Red Team               |
|                               |                           |
|                           Consensus                       |
|                               |                           |
|                      Deterministic Risk Engine            |
|                               |                           |
|                           Decision                        |
|                               |                           |
|                        Execution Gateway                  |
+-------------------------------+---------------------------+
                                |
                                v
+-----------------------------------------------------------+
|                    EVOLUTION SYSTEM                       |
|                                                           |
| Performance -> Attribution -> Agent Factory               |
|                                  |                        |
|                              Challengers                  |
|                                  |                        |
|             Historical -> OOS -> Walk Forward             |
|                                  |                        |
|                              Shadow Mode                  |
|                                  |                        |
|                       Champion/Challenger                 |
|                                  |                        |
|                      Promote / Retire / Rollback          |
+-----------------------------------------------------------+
```

---

# 4. Repository-Struktur

```text
.
├── README.md
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── .python-version
├── .env.example
├── config/
│   ├── risk-policy.yaml
│   └── population-policy.yaml
├── prompts/
│   ├── analysis/
│   ├── adversarial/
│   ├── decision/
│   └── evolution/
├── schemas/
├── src/trading_harness/
│   ├── api/
│   ├── llm/
│   ├── services/
│   ├── config.py
│   ├── main.py
│   └── models.py
├── tests/
└── docs/
```

---

# 5. Schnellstart

## Voraussetzungen

- Docker
- Docker Compose v2

```bash
./scripts/bootstrap.sh --docker
```

API:

```text
http://localhost:8080
```

Health Check:

```bash
curl http://localhost:8080/health
```

OpenAPI:

```text
http://localhost:8080/docs
```

---

# 6. Lokale Entwicklung

Voraussetzungen: Git und
[uv](https://docs.astral.sh/uv/getting-started/installation/) 0.11.x. `uv` installiert die
in `.python-version` festgelegte Python-Version bei Bedarf selbst. Der erste Befehl erzeugt eine
lokale `.env` aus der Vorlage und installiert exakt die in `uv.lock` gesperrten Abhängigkeiten:

```bash
./scripts/bootstrap.sh
make check
make run
```

Ein vollständiger lokaler Check ist auch direkt möglich:

```bash
./scripts/bootstrap.sh --check
```

## Entwicklung auf mehreren Geräten

Der reproduzierbare Übergabepunkt ist immer ein Git-Commit. Auf einem neuen Gerät genügt:

```bash
git clone https://github.com/novlazar23/smith.git
cd smith
./scripts/bootstrap.sh
make check
```

Vor dem Gerätewechsel Änderungen auf einem eigenen Branch committen und ausdrücklich zu GitHub
pushen; auf dem Zielgerät denselben Branch auschecken und `./scripts/bootstrap.sh` erneut ausführen.
`uv.lock`, `.python-version`, Konfigurationen, Schemas und Prompts gehören in Git. `.env`, `.venv`,
API-Schlüssel, Datenbankinhalte und Docker-Volumes bleiben absichtlich lokal und dürfen nicht
committet werden. Benötigt ein zweites Gerät denselben Datenbestand, muss dieser separat über einen
verschlüsselten Datenbank-Backup/Restore-Prozess übertragen werden.

CI verwendet ebenfalls die gesperrte Umgebung und führt `make check` aus. Damit wird derselbe
Test-, Lint- und Typprüfungs-Gate lokal und auf GitHub ausgeführt.

## Autarke Entwicklung mit OpenCode

OpenCode benötigt keine Codex- oder Harness-Installation. Nach Bootstrap und eigener
Provider-/Modell-Anmeldung wird es im Repository-Root gestartet:

```bash
./scripts/bootstrap.sh
opencode
```

`AGENTS.md` enthält die verbindlichen Projekt- und Sicherheitsregeln. `opencode.json` erlaubt
autonome Lese-, Editier-, Test- und Recherchearbeit im Repository, verlangt aber eine Bestätigung
für `git push` und blockiert Force-Push sowie `git reset --hard`. Die Modellwahl und Zugangsdaten
bleiben bewusst in der persönlichen OpenCode-Konfiguration und werden nicht in Git gespeichert.

Projektbefehle:

- `/resume` rekonstruiert den Stand ausschließlich aus Git und `docs/handoff.md` und setzt die
  Entwicklung fort.
- `/check` führt den vollständigen Qualitäts-Gate aus und behebt Fehler iterativ.
- `/handoff` prüft, dokumentiert und committet einen übergabefähigen Stand; ein Push benötigt eine
  ausdrückliche Freigabe.

---

# 7. LLM-Konfiguration

`.env`:

```text
LLM_BASE_URL=http://your-openai-compatible-gateway/v1
LLM_API_KEY=change-me
LLM_MODEL_FAST=local-fast
LLM_MODEL_MAIN=local-main
LLM_MODEL_CRITIC=local-critic
```

Die Modelle sind logische Profile. Die eigentliche Zuordnung darf außerhalb des Repositories über einen Gateway erfolgen.

---

# 8. Sicherheitsstatus des MVP

Aktuell:

```text
Live Execution: DISABLED
Paper/Shadow: vorgesehen
Risk Engine: vorhanden
Evolution Policy: vorhanden
Agent Registry: In-Memory MVP
Database: Infrastruktur vorbereitet
Redis: Infrastruktur vorbereitet
Exchange Adapter: absichtlich nicht implementiert
```

Das ist beabsichtigt.

Die erste produktive Ausbaustufe soll ausschließlich Shadow Trading durchführen.

---

# 9. Empfohlene Entwicklungsreihenfolge

## Phase 1 — Research Runtime

- persistente Agent Registry
- PostgreSQL
- Market Snapshot Store
- Agent Runtime
- LLM Gateway
- strukturierte Agent Outputs
- Trading Runs
- Performance Records

## Phase 2 — Evaluation

- Outcome Generator
- Brier Score
- Calibration
- MFE / MAE
- Expectancy
- Drawdown
- Regime Scoring
- Out-of-Sample
- Walk-Forward

## Phase 3 — Evolution

- Agent Factory
- Genome Persistenz
- Mutationen
- Recombination
- Challenger Pool
- Hall of Fame
- Graveyard
- Champion/Challenger
- Rollback

## Phase 4 — Paper Trading

- Paper Exchange
- Position Lifecycle
- deterministic Risk Engine
- Portfolio Risk
- Order Simulation
- Slippage

## Phase 5 — Live Execution

Erst nach stabiler Shadow-/Paper-Phase:

- separater Execution Service
- eigene Credentials
- Network Isolation
- Read/Trade API Trennung
- Kill Switch
- Rate Limits
- Order Dedupe
- minimaler Kapitaleinsatz

---

# 10. Grundprinzip

Das System optimiert nicht auf maximale Trade-Frequenz.

Es optimiert auf:

```text
robuste, kalibrierte, risikoadjustierte Entscheidungsqualität
über unbekannte Marktphasen hinweg
```

`NO_TRADE` ist eine vollwertige Entscheidung.
