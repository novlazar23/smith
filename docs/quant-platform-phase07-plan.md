# Arbeitsplan: Quant-Plattform Phase 7 — ML Features

> Phase 6 abgeschlossen (2026-08-25, `be72652`)

---

## Zusammenfassung

Phase 7 implementiert ML-Feature-Engineering — numerische Features für
Maschinelles Lernen aus den existierenden Quant-Modulen (Features, Anomalies, Regime,
Similarity, Forward Outcomes).

---

## Tasks

### P7-1: ML Feature Vector Builder
**Was:** `quant/ml_features.py` — Klasse `MLFeatureBuilder`:
- Kombiniert Features aus allen Quant-Modulen zu einheitlichem Vektor
- Normalisierung (Z-Score, Min-Max)
- Feature Selection (Korrelation, Importance)
- NaN-Handling

**Tests:** 10+ Unit-Tests

### P7-2: Feature Importance Engine
**Was:** `quant/feature_importance.py` — Klasse `FeatureImportanceEngine`:
- Berechnet Feature Importance via Korrelation, Mutual Information
- Feature-Ranking
- Feature-Group-Analyse

**Tests:** 8+ Unit-Tests

### P7-3: ML Features API Endpoints
**Was:** /quant/ml/features, /quant/ml/importance
**Tests:** 4+ neue Tests

### P7-4: Integration + Gate
**Tests:** 4+ Integration-Tests

### P7-5: Documentation + Handoff + Commit
