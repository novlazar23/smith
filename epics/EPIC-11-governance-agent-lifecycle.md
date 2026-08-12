# EPIC-11: Governance and Agent Lifecycle

## Problem
EPIC-01 bis EPIC-10 haben das komplette Trading-System gebaut. Jetzt fehlt die kontrollierte Steuerung der Agenten über ihren Lebenszyklus.

## Ziel
Kontrollierte Aktivierung, Überwachung, Herabstufung und Entfernung von Agenten mit:
- State Machine (SHADOW→ACTIVE, ACTIVE↔DEGRADED, ACTIVE→QUARANTINED)
- Promotion Rules (OOS, Kalibrierung, marginaler Nutzen, Review)
- Kill Criteria (automatische Degradation bei Drift/Fehlern)
- Champion-Challenger (Versionierung, Evaluation, Promotion)
- Audit Trail (Statusübergänge, Entscheidungen, Reviews)

## Abhängigkeiten
- EPIC-10 (Outcome Evaluation)
- EPIC-07 (Performance Reports, Historical Validation)
- EPIC-08 (Consensus, Dependency Analysis)

## Arbeitspakete

### WP01: Governance State Machine
- packages/governance/ — state_machine.py, base.py, blocking.py, engine.py
- State transitions: SHADOW→ACTIVE, ACTIVE↔DEGRADED, ACTIVE→QUARANTINED, QUARANTINED→SHADOW/DISABLED
- State factors: ACTIVE=1.0, DEGRADED=0.25-0.75, SHADOW/QUARANTINED/DISABLED=0.0
- Audit log of all transitions
- File: packages/governance/*.py

### WP02: Promotion Rules & Shadow Mode
- packages/governance/promotion_rules.py
- Neue Agenten immer SHADOW → final_weight == 0.0 in Consensus
- Aktivierung: Mindest-OOS, Kalibrierungsgrenze, positiver marginaler Nutzen
- Keine Schema-/Datenkritik, Review ok
- File: packages/governance/*.py

### WP03: Quarantine & Automatic Degradation
- packages/governance/quarantine/
- Automatische Quarantäne: Ungültige Ausgaben, Kalibrierungsverschlechterung, Drift
- Fehlende Evidenz, gestörte Quelle, unvert. Verteilung, Timeout
- Quarantäne → Gewicht 0.0 in Consensus
- File: packages/governance/quarantine/*.py

### WP04: Champion-Challenger System
- packages/governance/champion_challenger/
- Promotion: bessere OOS-Kalibrierung, gleich Stabilität, positiver marginaler Nutzen
- Keine neuen krit. Risiken, erfolgreicher Shadow-Betrieb
- Jede Variante: agent_id, champion_version, challenger_version, evaluation_window
- File: packages/governance/champion_challenger/*.py

### WP05: Audit & Kill Criteria
- packages/governance/audit/
- Audit Trail: Statusübergänge, Entscheidungen, Reviews
- Kill Criteria automatisiert auswertbar
- Rollback-Fähigkeit getestet
- File: packages/governance/audit/*.py

## DoD
- Neue Agenten immer SHADOW
- Shadow beeinflusst Entscheidung nicht
- Statusübergänge auditiert
- Automatische Degradation bei Drift/Fehlern
- Quarantäne → Gewicht 0.0
- Champion-Wechsel nachvollziehbar
- Rollback-Fähigkeit getestet
- Kill-Criteria automatisiert auswertbar
- Alle Tests bestanden, ruff clean

## Risiken
- Automatische Promotion ohne ausreichende Evidenz
- State Machine zu restriktiv/zu lax

## Rollback
- State Machine auf SHADOW zurücksetzen