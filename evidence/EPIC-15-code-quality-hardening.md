## EPIC-15 Code Quality Hardening — Verification Evidence

### DoD Checklist
- [x] 62 pyright errors fixed across 30 files
- [x] pyright packages/: 0 errors, 0 warnings
- [x] No TODO/FIXME/HACK/placeholder added in fixes
- [x] All changes are type fixes only, no behavioral changes
- [x] 5 semantic commits with descriptive messages

### Error Categories Fixed
- **WP01** (17 errors): Agent Report missing `expected_return`/`calibrated_confidence` — 12 agent files
- **WP02** (6 errors): `max(dict.get())` anti-pattern + numpy types — 2 files  
- **WP03** (11 errors): Backtesting type issues — 5 files
- **WP04** (3 errors): Governance state machine None-safety — 1 file
- **WP05** (17 errors): Observability + misc typing — 10 files

### pyright Verification
```
0 errors, 0 warnings, 0 informations
```

### Files Changed
30 files modified: 6 agents, 2 backtesting, 1 governance, 3 observability, 2 persistence, 2 orchestrator, 1 streaming, 1 uncertainty, 1 domain/news, 2 regime