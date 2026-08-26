# Arbeitsplan: Quant-Plattform Phase 3 — Anomaly Detection

> Phase 2 abgeschlossen (2026-08-25, `7f48d1f`)
> Leitprinzip: Phasen 2–8 berühren keinen existierenden Smith-Service

---

## Zusammenfassung

Phase 3 implementiert die Anomalieerkennung für OHLCV- und Feature-Daten.
Anomalien werden im `anomalies`-Measurement von InfluxDB gespeichert und über
API-Endpunkte abfragbar.

---

## Tasks

### P3-1: Anomaly Detection Engine
**Was:** `quant/anomaly_detection.py` — Klasse `AnomalyDetector`:
- Z-Score-basierte Erkennung (threshold=3.0)
- IQR-basierte Erkennung (1.5× IQR)
- Rolling-Window-Ansatz (configurable window)
- Detektiert: Preis-Schocks, Volumen-Spikes, Volatilitäts-Outliers

**Dateien:** `src/trading_harness/quant/anomaly_detection.py`, `tests/test_quant_anomaly_detection.py`
**Abhängigkeiten:** P2-1 (FeatureEngine — Volatilitäts-Features)
**Tests:** 10+ Unit-Tests

### P3-2: Anomaly Store
**Was:** `quant/anomaly_store.py` — Klasse `AnomalyStore`:
- `detect_and_store(symbols, timeframe, candles, exchange) -> AnomalyResult`
- `get_anomalies(symbol, timeframe, anomaly_type, start, end) -> list[dict]`
- Schreibt in `ANOMALY_MEASUREMENT` aus `schema.py`

**Dateien:** `src/trading_harness/quant/anomaly_store.py`, `tests/test_quant_anomaly_store.py`
**Abhängigkeiten:** P3-1, P1-4 (InfluxDBStore), P1-5 (schema)
**Tests:** 6+ Unit-Tests

### P3-3: Anomaly API Endpoints
**Was:** Erweiterung von `api/quant_routes.py`:
- `POST /quant/anomalies/detect` — Anomalien erkennen und speichern
- `GET /quant/anomalies/{symbol}` — Anomalien abfragen

**Dateien:** `src/trading_harness/api/quant_routes.py` (erweitern), `tests/test_quant_routes.py` (erweitern)
**Abhängigkeiten:** P3-2
**Tests:** 4+ neue Tests

### P3-4: Integration + Gate
**Was:** `tests/test_quant_anomaly_integration.py`
**Tests:** 4+ Integration-Tests
**Gate:** `make check` grün

### P3-5: Documentation + Handoff + Commit

---

## Parallelisierung

```
P3-1 (Anomaly Engine) ──┬── P3-2 (Anomaly Store) ──┬── P3-3 (API)
                         └── P3-4 (Integration) ────┘
```
