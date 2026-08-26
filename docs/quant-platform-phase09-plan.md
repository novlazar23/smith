# Arbeitsplan: Quant-Plattform Phase 9 — Shadow-Loop Integration

> Phase 8 abgeschlossen (2026-08-25, `e0120e4`)

---

## Zusammenfassung

Phase 9 integriert die Quant-Plattform vollständig in den Shadow Trading Loop.
Die existierende `shadow_trading_loop.py` wird erweitert, um die Quant-Module
(Features, Anomalien, Regime, Similarity, Forward Outcomes, ML Features, Backtesting)
als Evidence für den Trading-Orchestrator bereitzustellen.

**WICHTIG:** Nur minimale Änderungen an existierendem Code — die Integration
erfolgt primär über die Shadow-Loop-Hooks, die in Phase 1 (P1-7) bereits implementiert wurden.

---

## Tasks

### P9-1: Shadow-Loop Quant Integration
**Was:** Erweitert `shadow_trading_loop.py` um:
- Feature-Extraktion für jedes Symbol
- Anomalie-Erkennung pro Tick
- Regime-Erkennung pro Tick
- Zusammenfassung aller Quant-Evidence als strukturiertes Dict

**Tests:** 8+ Integration-Tests

### P9-2: Quant Evidence Aggregator
**Was:** `quant/evidence_aggregator.py` — Klasse `EvidenceAggregator`:
- Kombiniert alle Quant-Evidence zu einem einheitlichen Dict
- Priorisierung und Gewichtung
- Zeitstempel-Tracking

**Tests:** 6+ Unit-Tests

### P9-3: Shadow-Loop API Erweiterung
**Was:** Erweitert /quant/status um Shadow-Loop-Integration-Status
**Tests:** 3+ neue Tests

### P9-4: Integration + Gate
**Tests:** 4+ Integration-Tests

### P9-5: Documentation + Handoff + Commit
