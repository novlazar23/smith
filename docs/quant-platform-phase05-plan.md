# Arbeitsplan: Quant-Plattform Phase 5 — Similarity Engine

> Phase 4 abgeschlossen (2026-08-25, `00563ba`)

---

## Zusammenfassung

Phase 5 implementiert den Similarity Engine — findet historisch ähnliche Marktbedingungen
mittels DTW (Dynamic Time Warping) oder Euclidean Distance auf normalisierten OHLCV-Sequenzen.

---

## Tasks

### P5-1: Similarity Engine
**Was:** `quant/similarity.py` — Klasse `SimilarityEngine`:
- Euclidean Distance auf normalisierten Preissequenzen
- Sliding Window Matching
- Top-K ähnlichste historische Fenster
- Konfigurierbare Metrik und Window-Größe

**Tests:** 10+ Unit-Tests

### P5-2: Similarity Store
**Was:** `quant/similarity_store.py` — Klasse `SimilarityStore`
**Tests:** 5+ Unit-Tests

### P5-3: Similarity API Endpoints
**Was:** /quant/similarity/find, /quant/similarity/{symbol}
**Tests:** 4+ neue Tests

### P5-4: Integration + Gate
**Tests:** 4+ Integration-Tests

### P5-5: Documentation + Handoff + Commit
