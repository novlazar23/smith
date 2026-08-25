# Infrastruktur-Audit

> Phase 0 — Bestandsaufnahme  
> Erstellt: 2026-08-25

---

## 1. Host-System

| Ressource | Wert |
|-----------|------|
| CPU | 16 Kerne |
| RAM | 13 GiB total, ~8 GiB verfügbar |
| Storage | 1.8 TiB total, 201 GiB belegt (12%) |
| OS | Linux (Ubuntu, LVM) |
| Python | 3.12.3 |
| uv | 0.11.27 |
| Docker | Installiert, Compose v2 |
| Git | Vorhanden |

**Bewertung:** Ausreichend für InfluxDB + Feature-Engine. Kein GPU-Bedarf für Phase 0-1.

---

## 2. Docker-Services (smith docker-compose.yml)

| Service | Image | Port | Status |
|---------|-------|------|--------|
| `api` | Eigenes Build (Python 3.12-slim) | 8080 | Konfiguriert, nicht aktiv |
| `postgres` | timescale/timescaledb:latest-pg17 | intern | Konfiguriert |
| `redis` | redis:8-alpine | intern | Konfiguriert |

**Volumes:**
- `postgres-data` → `/var/lib/postgresql/data`
- `redis-data` → `/data`
- `./data` → `/app/data` (schreibbar, Kill-Switch + Execution-Log)

**Mounts (api-Service):**
- `./config:/app/config:ro`
- `./prompts:/app/prompts:ro`
- `./schemas:/app/schemas:ro`
- `./data:/app/data` (schreibbar)

**Bewertung:** Saubere Architektur. Keine Ports kollidieren mit InfluxDB (8086). Kein Volume-Name-Konflikt.

---

## 3. InfluxDB-Status

| Prüfpunkt | Status |
|-----------|--------|
| `influx` CLI | Nicht installiert |
| Docker Image | Nicht vorhanden |
| Config in Settings | Nicht vorhanden |
| Python Client | Nicht in pyproject.toml |

**Bewertung:** Grünfeld — kein bestehender InfluxDB, keine Migration nötig. Saubere Neuanlage möglich.

---

## 4. Python-Abhängigkeiten (Smith aktuell)

**Kern:**
- fastapi, uvicorn, pydantic, pydantic-settings, httpx, redis, psycopg, PyYAML

**Dev:**
- pytest, pytest-cov, ruff, mypy, pytest-asyncio

**Fehlend (für Quant-Plattform nötig):**
- `influxdb-client` — InfluxDB Python Client
- `numpy` — Numerische Berechnungen
- `pandas` / `polars` — Optionale Datenverarbeitung (später)
- `scikit-learn` — ML-Baselines (Phase 8)

**Bewertung:** Optionale Dependencies als `[quant]`-Extra hinzufügen (kein Pflicht-Dependency für Smith-Basis).

---

## 5. Ressourcen-Schätzung (Phase 1+)

### InfluxDB

| Komponente | Schätzung |
|------------|-----------|
| RAM (InfluxDB) | 512 MiB – 1 GiB (oss-2.7, 10 Symbole) |
| Storage (OHLCV/Jahr) | ~2.6 GiB (10 Symbole × 6 Timeframes × 50 Byte/Punkt) |
| Storage (Trades/Jahr) | ~5-10 GiB (hochauflösend, abhängig von Volumen) |
| CPU (Write) | <1 Kern (empfangene Punkte) |
| CPU (Query) | 1-2 Kerne (Feature-Berechnung) |

### Feature-Engine

| Komponente | Schätzung |
|------------|-----------|
| RAM (Batch) | 512 MiB – 2 GiB (Polars/DataFrame) |
| CPU (Berechnung) | 2-4 Kerne (EMA/RSI/ATR parallel) |
| Storage (Features) | ~500 MiB/Jahr (aggregierte Features) |

### Gesamt

| Ressource | Empfehlung |
|-----------|------------|
| RAM | 4 GiB reserviert (InfluxDB + Feature-Engine) |
| CPU | 4 Kerne reserviert |
| Storage | 20 GiB/Jahr (Daten + Features + Backups) |

**Bewertung:** Host hat 13 GiB / 16 Kerne. Platz für InfluxDB + Feature-Engine OHNE Beeinträchtigung der bestehenden Smith-Services.

---

## 6. Port-Belegung

| Port | Service |
|------|---------|
| 8080 | Smith API (FastAPI) |
| 5432 | PostgreSQL (intern) |
| 6379 | Redis (intern) |
| **8086** | **InfluxDB (geplant)** |

Keine Konflikte.

---

## 7. Sicherheitscheckliste

- [x] Kein `.env` im Repository
- [x] Docker-Mounts korrekt (nur `data/` schreibbar)
- [x] PostgreSQL/Redis ohne Host-Port-Mapping (intern)
- [x] InfluxDB: gleiche Strategie — intern, kein Host-Port außer Debug
- [x] API-Keys nicht im Code
- [x] `live_execution_enabled=False` (unverändert)
