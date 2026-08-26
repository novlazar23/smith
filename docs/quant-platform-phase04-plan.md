# Arbeitsplan: Quant-Plattform Phase 4 — Regime Detection

> Phase 3 abgeschlossen (2026-08-25, `2ef6999`)

---

## Zusammenfassung

Phase 4 implementiert die Regime-Erkennung (Marktphasen: bull/bear/range/high-vol/crash).
Regime werden im `regime`-Measurement von InfluxDB gespeichert.

---

## Tasks

### P4-1: Regime Detection Engine
**Was:** `quant/regime_detection.py` — Klasse `RegimeDetector`:
- Trend-basiert (SMA-Crossover, ADX)
- Volatilitäts-basiert (High/Low Vol)
- Crash/Recovery Erkennung
- Regime: strong_bull, weak_bull, range, weak_bear, strong_bear, high_volatility, low_volatility, crash, recovery

**Tests:** 10+ Unit-Tests

### P4-2: Regime Store
**Was:** `quant/regime_store.py` — Klasse `RegimeStore`
**Tests:** 5+ Unit-Tests

### P4-3: Regime API Endpoints
**Was:** /quant/regime/detect, /quant/regime/{symbol}
**Tests:** 4+ neue Tests

### P4-4: Integration + Gate
**Tests:** 4+ Integration-Tests

### P4-5: Documentation + Handoff + Commit
