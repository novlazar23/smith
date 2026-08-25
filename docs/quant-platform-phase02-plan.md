# Arbeitsplan: Quant-Plattform Phase 2 — Feature Engineering

> Phase 1 abgeschlossen (2026-08-25, `06f04f2`)
> Leitprinzip: Phasen 2–8 berühren keinen existierenden Smith-Service

---

## Zusammenfassung

Phase 2 implementiert die Feature-Berechnung (technische Indikatoren) aus OHLCV-Daten
und speichert die Ergebnisse im `features`-Measurement von InfluxDB. Alle Features
werden deterministisch aus den rohen Kerzen berechnet und versioniert (`FEATURE_VERSION`
aus `schema.py`).

---

## Tasks

### P2-1: Feature Computation Engine
**Was:** `quant/features.py` — Klasse `FeatureEngine` mit Indikatoren:
- RSI (14-period, Wilder-Smoothing)
- MACD (12/26/9)
- Bollinger Bands (20-period, 2σ)
- ATR (14-period)
- Volatility (20-period rolling std dev of log returns)
- VWAP (optional, wenn Volume vorhanden)

**Dateien:** `src/trading_harness/quant/features.py`, `tests/test_quant_features.py`
**Abhängigkeiten:** P1-6 (ohlcv_ingestion.py — Candle-Format)
**Tests:** 12+ Unit-Tests mit synthetischen Daten

### P2-2: Feature Storage
**Was:** `quant/feature_store.py` — Klasse `FeatureStore`:
- `compute_and_store(symbols, timeframe, candles) -> FeatureResult`
- `get_features(symbol, timeframe, feature_names, start, end) -> list[dict]`
- Schreibt in `FEATURE_MEASUREMENT` aus `schema.py`

**Dateien:** `src/trading_harness/quant/feature_store.py`, `tests/test_quant_feature_store.py`
**Abhängigkeiten:** P2-1, P1-4 (InfluxDBStore), P1-5 (schema)
**Tests:** 8+ Unit-Tests (mocked InfluxDB)

### P2-3: Feature API Endpoints
**Was:** Erweiterung von `api/quant_routes.py`:
- `POST /quant/features/compute` — Features berechnen und speichern
- `GET /quant/features/{symbol}` — Features abfragen

**Dateien:** `src/trading_harness/api/quant_routes.py` (erweitern), `tests/test_quant_routes.py` (erweitern)
**Abhängigkeiten:** P2-2
**Tests:** 5+ neue Tests

### P2-4: Integration + Gate
**Was:** `tests/test_quant_feature_integration.py` — Kompositionstests
**Tests:** 5+ Integration-Tests
**Gate:** `make check` grün, 980+ Tests

### P2-5: Documentation + Handoff
**Was:** handoff.md aktualisieren, Phase 2 Commit

---

## Parallelisierung

```
P2-1 (Feature Engine) ──┬── P2-2 (Feature Store) ──┬── P2-3 (API)
                         └── P2-4 (Integration) ────┘
```

P2-1 ist die einzige echte Abhängigkeit. P2-2+P2-3+P2-4 können parallel laufen
nach Abschluss von P2-1.

---

## Risiken

- **Performance:** RSI/MACD/BB sind O(n) pro Symbol — akzeptabel für MVP
- **Datenqualität:** Features benötigen mind. 26 Kerzen für MACD; weniger Kerzen → NaN
- **Speicher:** ~6 Features × 5 Fields × 1h = ~30 Punkte/Symbol/Tag = minimal
