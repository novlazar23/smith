# Phase 5 — Live Execution Epic

## Dependencies
- Phase 1 (Research Runtime) — ✅ COMPLETE
- Phase 2 (Evaluation) — ✅ COMPLETE
- Phase 3 (Evolution) — ✅ COMPLETE
- Phase 4 (Paper Trading) — ✅ COMPLETE
- Spec: `specs/phase-5-live-execution.md`
- Problem: `specs/phase-5-live-execution-problem.md`

## Warnung

**Live Execution ist ein sicherheitskritisches Feature.** Jedes implementierungs-Bug kann
finanziellen Schaden verursachen. Dieser Epic bleibt standardmäßig deaktiviert —
**keine Promotion vor Security Review**.

## Work Items

### WI-P5-1: KillSwitch

**Ziel**: Thread-sicherer Kill Switch mit persistiertem Zustand

**Eingaben**: Spec `specs/phase-5-live-execution.md`, R5.5-R5.8

**Dateien**:
- `src/trading_harness/services/kill_switch.py` — new
- `tests/test_kill_switch.py` — new

**Akzeptanz**:
- `activate()`, `deactivate()`, `is_active()` thread-safe
- Zustand persistiert (SQLite, fallback: in-memory)
- Aktivierung innerhalb von 100ms wirksam
- 10+ Tests, alle grün

### WI-P5-2: RateLimiter

**Ziel**: Token-Bucket Rate Limiter (global + per-symbol)

**Eingaben**: WI-P5-1, Spec `specs/phase-5-live-execution.md`, R5.9-R5.11

**Dateien**:
- `src/trading_harness/services/rate_limiter.py` — new
- `tests/test_rate_limiter.py` — new

**Akzeptanz**:
- Globaler Limit (Standard: 10/min) enforced
- Pro-Symbol Limit (Standard: 2/min) enforced
- Thread-safe unter concurrency
- 10+ Tests, alle grün

### WI-P5-3: OrderDeduplicator

**Ziel**: Thread-sichere Order-Dedup basierend auf decision_id + symbol + side

**Eingaben**: WI-P5-1, Spec `specs/phase-5-live-execution.md`, R5.12-R5.14

**Dateien**:
- `src/trading_harness/services/order_deduplicator.py` — new
- `tests/test_order_deduplicator.py` — new

**Akzeptanz**:
- Duplikate zu 100% erkannt (getestet mit parallelen Calls)
- Thread-safe
- Memory-bounded (kein unendliches Wachsen)
- 10+ Tests, alle grün

### WI-P5-4: ExchangeAdapter Interface

**Ziel**: Polymorphes Exchange-Adapter Interface (kein konkreter Exchange im MVP)

**Eingaben**: Spec `specs/phase-5-live-execution.md`, R5.4

**Dateien**:
- `src/trading_harness/services/exchange_adapter.py` — new (interface + stub)
- `tests/test_exchange_adapter.py` — new

**Akzeptanz**:
- `submit_order(proposal) -> dict` Interface definiert
- Stub-Implementierung gibt `NOT_IMPLEMENTED` zurück
- Erweiterbar für CCXT-Integration in Zukunft
- 5+ Tests, alle grün

### WI-P5-5: LiveExecutionService

**Ziel**: Orchestriert KillSwitch → RateLimiter → Deduplicator → ExchangeAdapter

**Eingaben**: WI-P5-1, WI-P5-2, WI-P5-3, WI-P5-4, Spec `specs/phase-5-live-execution.md`, R5.1-R5.3, R5.23-R5.24

**Dateien**:
- `src/trading_harness/services/live_execution_service.py` — new
- `tests/test_live_execution_service.py` — new

**Akzeptanz**:
- Full Flow: validate → kill switch → rate limit → dedupe → exchange → log
- Execution log mit decision_id, timestamp, result
- Standardmäßig deaktiviert (live_enabled=false blockiert alles)
- Min Capital enforcement
- 20+ Tests, alle grün

### WI-P5-6: ExecutionLog Store

**Ziel**: Persistente Execution Logs mit In-Memory-Fallback

**Eingaben**: WI-P5-5, Spec `specs/phase-5-live-execution.md`, R5.17 (Network Policy), R5.20

**Dateien**:
- `src/trading_harness/services/execution_store.py` — new
- `src/trading_harness/services/db.py` — extend INIT_SQL
- `tests/test_execution_store.py` — new

**Akzeptanz**:
- execution_log Tabelle
- In-Memory-Fallback
- Credentials nie in Logs
- 10+ Tests, alle grün

### WI-P5-7: API Routes

**Ziel**: Execution API Endpoints mit Read/Trade API Separation

**Eingaben**: WI-P5-5, WI-P5-6, Spec `specs/phase-5-live-execution.md`, R5.21-R5.22

**Dateien**:
- `src/trading_harness/api/routes.py` — extend
- `tests/test_api_execution.py` — new

**Akzeptanz**:
- POST /execution/orders — Order submit (auth required)
- POST /execution/kill-switch — Kill Switch toggle (auth required)
- GET /execution/status — Execution Status
- GET /execution/logs — Execution Logs (read auth)
- 15+ Tests, alle grün

### WI-P5-8: Dokumentation & Cleanup

**Ziel**: handoff.md aktualisieren, docs aktualisieren, Security Review vorbereiten

**Eingaben**: Alle vorherigen WIs

**Dateien**:
- `docs/handoff.md` — update
- `README.md` — update Phase 5 status
- `docs/security-review-phase5.md` — new (Security Review Checklist)

**Akzeptanz**:
- handoff.md: Phase 5 als COMPLETE markiert
- Security Review Checklist erstellt
- make check clean
- **Live Execution standardmäßig deaktiviert**

## Critical Path

WI-P5-1 → WI-P5-5 → WI-P5-6 → WI-P5-7 → WI-P5-8
WI-P5-2 ──┘
WI-P5-3 ──┘
WI-P5-4 ──┘

## Parallelisierung

- WI-P5-1, WI-P5-2, WI-P5-3, WI-P5-4 können parallel starten (unabhängig)
- WI-P5-5 benötigt WI-P5-1, WI-P5-2, WI-P5-3, WI-P5-4
- WI-P5-6 kann parallel zu WI-P5-5 starten (unabhängige Persistenz)
- WI-P5-7 benötigt WI-P5-5, WI-P5-6
- WI-P5-8 kann parallel zu WI-P5-7 starten (dokumentation-only)

## Risiken

- **R1** Kritisches Risiko: Live Execution mit Fehlern kann finanziellen Schaden verursachen.
  → Abhilfe: Standardmäßig deaktiviert, keine Promotion vor Security Review.

- **R2** Kill Switch Zuverlässigkeit: muss deterministisch und sofort wirken.
  → Abhilfe: Thread-safe mit Lock, <100ms Guarantee, Tests mit concurrency.

- **R3** Credential Exposure: muss in Logs/Audit vermieden werden.
  → Abhilfe: Blacklist in Logging, Tests die Credential-Exposure prüfen.

- **R4** Race Conditions: Duplicate Orders bei paralleler Ausführung.
  → Abhilfe: Dedup mit Lock, atomare Checks.

## Definition of Done

- Alle WIs abgeschlossen
- make check clean (tests, lint, mypy)
- docs/handoff.md aktualisiert
- Security Review Checklist erstellt
- Live Execution standardmäßig deaktiviert
- Kein spezifischer Exchange integriert (nur Interface)