# Arbeitsplan: Quant-Plattform Phase 0+1

> Integration in das Smith Trading Harness  
> Strategie B (Snapshot-Enrichment)  
> Erstellt: 2026-08-25

---

## Zusammenfassung

Phase 0 (Bestandsaufnahme) und Phase 1 (InfluxDB + OHLCV-Ingestion) bereiten die
Infrastruktur für eine modulare Quantitative Trading-Daten- und Analyseplattform vor,
die als eigenständiges Modul (`src/trading_harness/quant/`) im Smith-Repo lebt, ohne
bestehenden Code zu verändern.

**Leitprinzip:** Die ersten 8 Phasen (0–8) berühren keinen existierenden Smith-Service.
Integration in den Shadow-Loop beginnt erst Phase 9.

---

## Phase 0: Bestandsaufnahme + Architektur (KEIN Code)

### P0-1: Docker-Infrastruktur-Audit
**Was:** Prüfen welche Services laufen, Ressourcenverbrauch, Port-Belegung, Volume-Mounts.
**Dateien:** `docker-compose.yml`, `Dockerfile`
**Ergebnis:** Dokumentation in `docs/infrastructure-audit.md`
**Risiko:** Gering — rein lesend
**Parallelisierbar:** Ja (mit P0-2, P0-3)
**Definition of Done:** Infra-Dokument existiert mit: laufende Services, Ports, Volumes, Ressourcen-Schätzung

### P0-2: Marktdatenfluss-Analyse
**Was:** Wie gelangen Ticker vom Exchange-Adapter zum Shadow-Loop? Welche Protokolle/Interfaces existieren?
**Betroffene Dateien:**
- `src/trading_harness/services/shadow_trading_loop.py` (MarketDataProvider Protocol, Zeile 93)
- `src/trading_harness/services/crypto_exchange_adapter.py`
- `src/trading_harness/services/paper_exchange_adapter.py`
- `src/trading_harness/services/shadow_trading_service.py`
**Ergebnis:** Flussdiagramm + Interface-Definition in `docs/quant-platform-data-flow.md`
**Risiko:** Gering — rein lesend
**Parallelisierbar:** Ja
**Definition of Done:** Dokument mit: (1) MarketDataProvider-Protokoll, (2) CryptoMarketDataProvider-Wrapper, (3) Shadow-Loop-Aufrufkette, (4) Integrationspunkte für Quant-Engine markiert

### P0-3: InfluxDB-Versionierung
**Was:** InfluxDB 2.x OSS vs. 3.x (Apache Arrow/DataFusion) evaluieren.
**Entcheidungskriterien:**
- OSS-Kosten vs. Cloud
- Python-Client-Reife (`influxdb-client` vs. `influxdb-client-python`)
- Downsampling-Support (Kapazity-Task vs. Continuous Query)
- Community-Stabilität, Docker-Image-Qualität
- Speicher-Effizienz für hochauflösende Crypto-Daten
**Ergebnis:** ADR (Architecture Decision Record) in `docs/adr-001-influxdb-version.md`
**Risiko:** Mittel — falsche Wahl verursacht Migration
**Parallelisierbar:** Ja
**Definition of Done:** ADR mit: Empfehlung, Begründung, Alternativen, Risiken, Reversal-Strategie

### P0-4: Ressourcen-Schätzung
**Was:** CPU/RAM/Storage für InfluxDB + Feature-Engine schätzen.
**Annahmen:**
- 10 Symbole × 6 Timeframes × 1m-Granularität
- ~14.400 OHLCV-Punkte/Symbol/Tag (1m)
- ~144.000 Punkte/Tag gesamt (10 Symbole)
- Speicher: ~50 Byte/Punkt × 144k × 365 Tage = ~2.6 GB/Jahr (OHLCV allein)
- Feature-Berechnung: ~2 CPU-Kerne für Batch-Updates
**Ergebnis:** Ressourcen-Tabelle in `docs/infrastructure-audit.md`
**Risiko:** Gering — Schätzung
**Parallelisierbar:** Ja
**Definition of Done:** Ressourcen-Empfehlung mit RAM/CPU/Storage-Disk-Schätzung

### P0-5: Integrationsgrenzen definieren
**Was:** Festlegen was Phase 0+1 ändert und was NICHT.
**Status:** ✅ Komplett (2026-08-25)
**Change-Set Phase 0+1:**

**GEÄNDERT (Phase 0 — nur Doku):**
| Datei | Aktion |
|-------|--------|
| `docs/infrastructure-audit.md` | NEU — Host-Ressourcen, Docker-Services, Ressourcen-Schätzung |
| `docs/quant-platform-data-flow.md` | NEU — Datenfluss-Analyse, Integrationspunkte |
| `docs/adr-001-influxdb-version.md` | NEU — InfluxDB 2.7 vs 3.x Entscheidung |
| `docs/quant-platform-phase01-plan.md` | NEU — Dieser Arbeitsplan |

**GEÄNDERT (Phase 1 — Code + Infra):**
| Datei | Aktion |
|-------|--------|
| `docker-compose.yml` | NEUER Service `influxdb` (InfluxDB 2.7 OSS) |
| `pyproject.toml` | NEUES optional dependency group `[quant]` |
| `src/trading_harness/config.py` | 5 neue Settings-Felder (influxdb_*) |
| `src/trading_harness/quant/__init__.py` | NEU — Quant-Modul-Package |
| `src/trading_harness/quant/influxdb_client.py` | NEU — InfluxDB Client Wrapper |
| `src/trading_harness/quant/schema.py` | NEU — Measurement/Tag/Field-Definitionen |
| `src/trading_harness/quant/ohlcv_ingestion.py` | NEU — OHLCV-Ingestion + Downsampling |
| `src/trading_harness/quant/observability.py` | NEU — Metriken + Health |
| `src/trading_harness/api/quant_routes.py` | NEU — API-Endpunkte |
| `tests/test_quant_config.py` | NEU — Config-Tests |
| `tests/test_quant_influxdb_client.py` | NEU — Client-Tests |
| `tests/test_quant_ohlcv.py` | NEU — Ingestion-Tests |
| `tests/test_quant_api.py` | NEU — API-Tests |
| `tests/test_quant_integration.py` | NEU — Integration-Tests (mit InfluxDB) |

**EINZIGE ÄNDERUNG an bestehendem Code:**
| Datei | Zeile | Änderung |
|-------|-------|----------|
| `shadow_trading_loop.py` | ~Init | +1 optionaler Parameter `ohlcv_ingestor` |
| `shadow_trading_loop.py` | ~run_once() | +2 Zeilen: `if self._ohlcv_ingestor: await ...` |

**NICHT GEÄNDERT ( explizit ausgeschlossen ):**
| Datei | Grund |
|-------|-------|
| `src/trading_harness/services/shadow_trading_service.py` | Kein Behavior-Change |
| `src/trading_harness/api/routes.py` | Neue Datei `quant_routes.py` stattdessen |
| `src/trading_harness/main.py` | Integration erst Phase 9 |
| `src/trading_harness/services/crypto_exchange_adapter.py` | Adapter bleiben unberührt |
| `src/trading_harness/services/risk_engine.py` | Risk-Engine unverändert |
| `src/trading_harness/services/kill_switch.py` | Kill-Switch unverändert |
| Alle 899 bestehenden Tests | Regression-Test |
| `.env` / `.env.example` | Keine Secrets |
**Ergebnis:** Change-Set-Dokumentation in `docs/quant-platform-phase01-plan.md`
**Risiko:** Gering
**Parallelisierbar:** Nein (letzter P0-Task)
**Definition of Done:** Klare Dateiliste mit +/- Markierung

---

### Phase 0 Abnahmegate
```
□ infrastructure-audit.md existiert
□ quant-platform-data-flow.md existiert
□ adr-001-influxdb-version.md existiert, approved
□ Ressourcen-Schätzung dokumentiert
□ Change-Set definiert
□ Kein einziger Test geändert
□ make check nach Phase 0 = 899 passed
```

---

## Phase 1: InfluxDB + OHLCV-Ingestion

### Abhängigkeiten
- Phase 0 komplett (ADR approved, Change-Set definiert)
- Keine Abhängigkeiten zwischen Tasks innerhalb von Phase 1 (außer P1-3 hängt von P1-2 ab)

---

### P1-1: InfluxDB Service zu docker-compose.yml hinzufügen
**Was:** Neuer `influxdb` Service mit Health-Check, Volume, Netzwerk.
**Datei:** `docker-compose.yml`
**Spezifikation:**
```yaml
influxdb:
  image: influxdb:2.7
  restart: unless-stopped
  ports:
    - "8086:8086"
  environment:
    DOCKER_INFLUXDB_INIT_MODE: setup
    DOCKER_INFLUXDB_INIT_USERNAME: admin
    DOCKER_INFLUXDB_INIT_PASSWORD: admin-password-change-me
    DOCKER_INFLUXDB_INIT_ORG: smith
    DOCKER_INFLUXDB_INIT_BUCKET: market_data
    DOCKER_INFLUXDB_INIT_ADMIN_TOKEN: influxdb-token-change-me
  volumes:
    - influxdb-data:/var/lib/influxdb2
    - influxdb-config:/etc/influxdb2
  healthcheck:
    test: ["CMD", "influx", "ping"]
    interval: 10s
    timeout: 5s
    retries: 5
```
**Voraussetzungen:** `.env` muss `INFLUXDB_URL`, `INFLUXDB_TOKEN`, `INFLUXDB_ORG`, `INFLUXDB_BUCKET` enthalten
**Tests:** `docker compose config --quiet` muss exit 0
**Risiko:** Gering — isolierte Änderung
**Parallelisierbar:** Nein (muss vor P1-2)
**Definition of Done:** InfluxDB startet, Health-Check grün, `/ping` antwortet

### P1-2: Dependencies in pyproject.toml
**Was:** `influxdb-client` als optionale Abhängigkeit + `numpy` für Feature-Berechnung.
**Datei:** `pyproject.toml`
**Änderung:**
```toml
[project.optional-dependencies]
quant = [
  "influxdb-client>=1.40,<2",
  "numpy>=2.0,<3",
]
dev = [
  # ... existing ...
  "pytest-asyncio>=1.4.0",
]
```
**Begründung:** Quant-Dependencies als optionales Extra (`pip install -e ".[quant]"`), damit Basis-Smith ohne numpy/influxdb bleibt.
**Tests:** `uv sync --all-extras` muss funktionieren; `uv run pytest -q` = 899 passed
**Risiko:** Gering
**Parallelisierbar:** Ja (mit P1-1)
**Definition of Done:** `uv sync --all-extras` sauber, bestehende Tests unberührt

### P1-3: Config-Erweiterung
**Was:** InfluxDB-Konfiguration in Settings.
**Datei:** `src/trading_harness/config.py`
**Neue Felder:**
```python
# Quant Platform — InfluxDB
influxdb_url: str = "http://localhost:8086"
influxdb_token: str = ""
influxdb_org: str = "smith"
influxdb_bucket: str = "market_data"
influxdb_enabled: bool = False  # default-off, wie Shadow Trading
```
**Tests:** Unit-Test für Settings-Defaults (neuer Test in `tests/test_quant_config.py`)
**Risiko:** Gering — additive Änderung
**Parallelisierbar:** Nach P1-2
**Definition of Done:** `get_settings().influxdb_url` liefert Default; `influxdb_enabled=False`

### P1-4: InfluxDB Client Wrapper
**Was:** Thread-sicheren Client-Wrapper für InfluxDB-Operationen implementieren.
**Neue Datei:** `src/trading_harness/quant/influxdb_client.py`
**Verantwortlichkeiten:**
- `InfluxDBStore` Klasse mit `write_points()`, `query()`, `health_check()`
- Thread-Safety via `threading.RLock` (wie bestehende Stores)
- Fallback auf in-Memory-Store wenn InfluxDB nicht erreichbar (Pattern aus `db.py`)
- Kein `as any` oder Type-Suppression
**Protokoll:**
```python
class InfluxDBStore:
    def __init__(self, url: str, token: str, org: str, bucket: str): ...
    async def health_check(self) -> bool: ...
    async def write_points(self, measurement: str, tags: dict, fields: dict, timestamp: int) -> None: ...
    async def query(self, flux: str) -> list[dict]: ...
    def is_available(self) -> bool: ...  # True wenn InfluxDB erreichbar
```
**Tests:** `tests/test_quant_influxdb_client.py`
- Unit-Tests mit Mock-Client (kein echtes InfluxDB für Unit-Tests)
- Integration-Test mit echtem InfluxDB (Marker `real_influxdb`, opt-out via conftest)
**Risiko:** Mittel — Thread-Safety + Fallback-Pattern
**Parallelisierbar:** Nach P1-3
**Definition of Done:** 8+ Tests, ruff clean, mypy clean, Fallback funktioniert

### P1-5: InfluxDB Schema-Design
**Was:** Flux-Includes/Measurements für OHLCV, Trades, Orderbook, Derivate definieren.
**Datei:** `src/trading_harness/quant/schema.py`
**Measurements:**
```python
# OHLCV
OHLCV_MEASUREMENT = "ohlcv"
OHLCV_TAGS = ["symbol", "exchange", "timeframe"]
OHLCV_FIELDS = ["open", "high", "low", "close", "volume"]

# Trades
TRADES_MEASUREMENT = "trades"
TRADES_TAGS = ["symbol", "exchange"]
TRADES_FIELDS = ["price", "size", "side"]

# Orderbook
ORDERBOOK_MEASUREMENT = "orderbook"
ORDERBOOK_TAGS = ["symbol", "exchange"]
ORDERBOOK_FIELDS = ["best_bid", "best_ask", "spread", "bid_depth", "ask_depth", "imbalance"]

# Derivate
DERIVATIVES_MEASUREMENT = "derivatives"
DERIVATIVES_TAGS = ["symbol", "exchange"]
DERIVATIVES_FIELDS = ["funding_rate", "open_interest", "open_interest_change",
                      "liquidations_long", "liquidations_short", "basis"]

# Anomalien (später Phase 3)
ANOMALY_MEASUREMENT = "anomalies"
ANOMALY_TAGS = ["symbol", "exchange", "anomaly_type"]
ANOMALY_FIELDS = ["anomaly_score", "severity", "feature"]
```
**Timeframes:** 1m, 5m, 15m, 1h, 4h, 1d
**Tests:** Schema-Validierungstests (Konstanten korrekt, Tags/Fields konsistent)
**Risiko:** Gering — reine Daten-Definition
**Parallelisierbar:** Ja (mit P1-4)
**Definition of Done:** Schema-Module existieren, Konstanten getestet

### P1-6: OHLCV-Ingestion von Crypto-Adapteln
**Was:** Bestehende `get_ticker()`-Antworten in InfluxDB-OHLCV-Punkte umwandeln.
**Neue Datei:** `src/trading_harness/quant/ohlcv_ingestion.py`
**Verantwortlichkeiten:**
- `OHLCVIngestor` Klasse: akzeptiert Ticker-Daten, schreibt OHLCV-Punkte
- Downsampling: 1m-Rohdaten → 5m/15m/1h/4h/1d (Batch-Task, nicht pro Tick)
- Use existing `CryptoMarketDataProvider.get_ticker()` als Datenquelle
- Kein Look-ahead Bias: Features nur aus vergangenen Daten berechnen
**Protokoll:**
```python
class OHLCVIngestor:
    def __init__(self, influx_store: InfluxDBStore, market_data: MarketDataProvider): ...
    async def ingest_ticker(self, symbol: str, exchange: str, timestamp: int) -> None: ...
    async def downsample(self, symbol: str, exchange: str) -> None: ...
    async def backfill(self, symbol: str, exchange: str, start: int, end: int) -> None: ...
```
**Tests:** `tests/test_quant_ohlcv_ingestion.py`
- Mock-InfluxDB-Store (kein echtes InfluxDB)
- Ticker→Point-Konvertierung
- Downsampling-Logik (1m→5m Aggregation korrekt)
- Fallback bei InfluxDB-Ausfall
**Risiko:** Mittel — Downsampling-Logik muss korrekt sein
**Parallelisierbar:** Nach P1-5
**Definition of Done:** 10+ Tests, ruff/mypy clean, Downsampling validiert

### P1-7: Shadow-Loop-Integration (OHLCV-Hook)
**Was:** OHLCV-Ingestion als optionalen Hook in den Shadow-Loop einhängen.
**Geänderte Datei:** `src/trading_harness/services/shadow_trading_loop.py`
**Änderung (minimal):**
- Neuer optionaler Parameter `ohlcv_ingestor: OHLCVIngestor | None = None` im `__init__`
- Nach `run_once()` Aufruf: `if self._ohlcv_ingestor: await self._ohlcv_ingestor.ingest_ticker(...)`
- Keine Änderung an bestehender Kette (Ticker→Snapshot→...→Paper)
**WICHTIG:** Dies ist die EINZIGE Änderung an bestehendem Code in Phase 1.
**Tests:**
- Unit-Test: OHLCV-Hook wird aufgerufen wenn konfiguriert
- Unit-Test: OHLCV-Hook wird NICHT aufgerufen wenn None (Default)
- Alle 899 Bestands-Tests bleiben grün (Regression)
**Risiko:** Hoch — touching existing loop code
**Gegenmaßnahme:** Nur 1 Zeile Code change, null Behavior-Änderung für alle die keinen Ingestor konfigurieren
**Parallelisierbar:** Nein (nach P1-6)
**Definition of Done:** Hook funktioniert, alle 899+ Tests grün, kein Behavior-Change für Default-Konfiguration

### P1-8: API-Endpunkte für Ingestion-Kontrolle
**Was:** Neue API-Endpunkte für InfluxDB-Ingestion.
**Neue Datei:** `src/trading_harness/api/quant_routes.py`
**Endpunkte:**
```
POST /quant/ingest/ohlcv     — Manuelle OHLCV-Ingestion für Symbol+Exchange+Zeitraum (Trade-Key)
GET  /quant/status            — InfluxDB-Health + Ingestionsstatistiken (Read-Key)
POST /quant/ingest/backfill   — Historische Daten von Exchange-API nachladen (Trade-Key)
GET  /quant/schema            — Verfügbare Measurements/Tags/Fields (Read-Key)
```
**Tests:** `tests/test_quant_api.py`
- Health-Check (InfluxDB erreichbar + nicht erreichbar)
- Schema-Response korrekt
- Auth-Trennung (Trade/Read Key)
**Risiko:** Gering — neue Datei, keine Konflikte
**Parallelisierbar:** Ja (mit P1-7)
**Definition of Done:** 6+ API-Tests, OpenAPI-Dokument korrekt, Auth getestet

### P1-9: Observability für InfluxDB
**Was:** Metriken/Health-Check für InfluxDB-Pipeline.
**Neue Datei:** `src/trading_harness/quant/observability.py`
**Metriken:**
- `influxdb_write_count` (pro Measurement)
- `influxdb_write_latency_ms`
- `influxdb_query_latency_ms`
- `influxdb_error_count`
- `ohlcv_ingestion_lag_seconds` (letzter Ingested-Timestamp vs. jetzt)
- `downsample_progress` (letzter downsampled-Timeframe)
**Tests:** Unit-Tests für Metriken-Sammlung
**Risiko:** Gering
**Parallelisierbar:** Ja (mit P1-8)
**Definition of Done:** Metriken-Objekt getestet,Health-Endpoint liefert Status

### P1-10: Integration Tests mit echtem InfluxDB
**Was:** End-to-End-Tests die tatsächlich InfluxDB starten und schreiben/lesen.
**Datei:** `tests/test_quant_integration.py`
**Marker:** `real_influxdb` (opt-out via conftest)
**Tests:**
1. InfluxDB-Health-Check (Start → Ping → OK)
2. OHLCV-Schreiben + Lesen (Roundtrip)
3. Downsampling (1m → 5m korrekte Aggregation)
4. Fallback (InfluxDB gestoppt → In-Memory-Modus)
5. Backfill (historische Daten einfügen)
**Risiko:** Mittel — braucht laufendes InfluxDB
**Parallelisierbar:** Nein (nach P1-4 bis P1-9)
**Definition of Done:** 5+ Integration-Tests, alle mit echtem InfluxDB grün

### P1-11: Dokumentation + Handoff
**Was:** README aktualisieren, Handoff updaten, Architektur-Doku.
**Dateien:**
- `README.md` — neuer Abschnitt „Quant Platform" unter Phase 0+1
- `docs/handoff.md` — Status-Update für Phase 0+1
- `docs/architecture.md` — InfluxDB-Integration dokumentieren
**Risiko:** Gering
**Parallelisierbar:** Nach P1-10
**Definition of Done:** README hat Quant-Sektion, Handoff aktualisiert

---

### Phase 1 Abnahmegate
```
□ docker compose config --quiet = exit 0
□ InfluxDB startet und antwortet auf /ping
□ uv sync --all-extras funktioniert
□ influxdb_enabled=False als Default (kein Behavior-Change)
□ OHLCV-Ingestion schreibt korrekte Punkte
□ Downsampling validiert (1m→5m)
□ Shadow-Loop-Hook funktioniert (Default: None → kein Behavior-Change)
□ API-Endpunkte funktional und getestet
□ Observability-Metriken vorhanden
□ Integration-Tests grün (mit echtem InfluxDB)
□ make check = 899+ Tests, ruff clean, mypy clean
□ Kein bestehender Test fehlschlägt
□ Kein .env-File committet
□ Dokumentation aktuell
```

---

## Risiko-Bewertung

| Task | Risiko | Grund | Gegenmaßnahme |
|------|--------|-------|---------------|
| P0-* | Gering | Rein lesend | — |
| P1-1 | Gering | Isolierte Docker-Änderung | Health-Check |
| P1-2 | Gering | Additive Dependencies | Optional-Extra |
| P1-3 | Gering | Neue Settings-Felder | Defaults + Tests |
| P1-4 | Mittel | Thread-Safety + Fallback | RLock + Memory-Fallback |
| P1-5 | Gering | Reine Daten-Definition | — |
| P1-6 | Mittel | Downsampling-Logik | Konkrete Aggregationstests |
| P1-7 | **Hoch** | Touching existing loop | 1-Zeile change, Default=None |
| P1-8 | Gering | Neue Datei | — |
| P1-9 | Gering | Metriken-Objekt | — |
| P1-10 | Mittel | Braucht InfluxDB | Marker für opt-out |
| P1-11 | Gering | Nur Doku | — |

---

## Parallelisierungs-Graph

```
Phase 0 (parallel):
  P0-1 ──┐
  P0-2 ──┤
  P0-3 ──┼──→ P0-5 → Gate 0
  P0-4 ──┘

Phase 1 (sequenziell mit parallelen Blöcken):
  P1-1 ──┐
  P1-2 ──┤ (parallel)
          ↓
  P1-3 → P1-4 ──┐
  P1-5 ──────────┤ (parallel)
                  ↓
  P1-6 → P1-7 ──┐
  P1-8 ──────────┤ (parallel)
  P1-9 ──────────┘
                  ↓
  P1-10 → P1-11 → Gate 1
```

**Geschätzte Dauer:**
- Phase 0: 1-2 Stunden (Doku + ADR)
- Phase 1: 6-10 Stunden (Implementation + Tests)
- Gesamt: 8-12 Stunden

---

## Nicht getan ( explizit ausgeschlossen )

- ✗ Keine Änderung an `shadow_trading_loop.py` Ketten-Logik
- ✗ Keine Änderung an `routes.py` (neue Datei `quant_routes.py`)
- ✗ Keine Änderung an `main.py` (Quant-Routes werden in Phase 9 eingehängt)
- ✗ Keine ML-Bibliotheken (erst Phase 8)
- ✗ Keine Feature-Engine (erst Phase 2)
- ✗ Keine Anomalie-Erkennung (erst Phase 3)
- ✗ Keine Regime-Detektion (erst Phase 4)
- ✗ Keine Similarity-Engine (erst Phase 5)
- ✗ Kein Live-Trading (niemals ohne explizite Freigabe)
