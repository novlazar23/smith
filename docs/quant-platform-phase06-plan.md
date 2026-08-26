# Arbeitsplan: Quant-Plattform Phase 6 — Forward Outcomes

> Phase 5 abgeschlossen (2026-08-25, `91bf280`)

---

## Zusammenfassung

Phase 6 implementiert Forward Outcome Statistics — berechnet was nach einem bestimmten
Markt-Muster tatsächlich passiert ist (returns nach N Kerzen, Hit Rate, Profit Factor).

---

## Tasks

### P6-1: Forward Outcomes Engine
**Was:** `quant/forward_outcomes.py` — Klasse `ForwardOutcomeEngine`:
- Berechnet Forward Returns nach einem Muster
- Hit Rate (wie oft positiv nach N Kerzen)
- Profit Factor, Expectancy
- Konfigurierbare Horizonte (5, 10, 20, 50 Kerzen)

**Tests:** 10+ Unit-Tests

### P6-2: Forward Outcomes Store
**Was:** `quant/forward_outcomes_store.py`
**Tests:** 4+ Unit-Tests

### P6-3: Forward Outcomes API Endpoints
**Was:** /quant/outcomes/compute, /quant/outcomes/{symbol}
**Tests:** 4+ neue Tests

### P6-4: Integration + Gate
**Tests:** 4+ Integration-Tests

### P6-5: Documentation + Handoff + Commit
