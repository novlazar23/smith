# WI-ST-06 — Shadow-Trading API & Wiring: Verifikationsnachweis

Datum: 2026-08-24 | Actor: Sisyphus | Workitem: WI-ST-06 (API & Wiring)

## Umfang der Implementierung

1. Neu: `src/trading_harness/services/shadow_trading_service.py` — `ShadowTradingService`-Fassade:
   Assembliert den `ShadowTradingLoop` aus verdrahteten Singletons (ShadowExecutionBackend auf dem
   Paper-Stack, CryptoMarketDataProvider über den echten Crypto-Router, TradingRunService,
   SnapshotStore/PersistedSnapshotStore, RiskEngine, KillSwitch, AgentGenomeStore, AgentRuntime,
   ShadowTradingStateStore aus `settings.shadow_state_path`). Operationen: `start()`, `stop()`,
   `run_once()`, `status()`, `records(symbol, status, limit)`, `portfolio(limit)`, `shutdown()`.
   Kein Autostart (Z2): `shutdown()` stoppt ausschließlich einen RUNNING-Loop.
2. `src/trading_harness/api/routes.py` — Singleton `shadow_trading_service` + 6 Endpunkte:
   - POST /shadow-trading/start (require_trade_key; Guards -> 403, ALREADY_RUNNING -> 409, detail.code maschinenlesbar)
   - POST /shadow-trading/stop (require_trade_key; idempotent)
   - POST /shadow-trading/run-once (require_trade_key)
   - GET /shadow-trading/status (require_read_key)
   - GET /shadow-trading/records?symbol=&status=&limit>=1 (require_read_key; limit = neueste N, aufsteigend)
   - GET /shadow-trading/portfolio?limit>=1 (require_read_key; current/start equity + M2M-Historie)
3. `src/trading_harness/main.py` — FastAPI-Lifespan: kein Autostart (Z2); beim Shutdown wird
   `await routes.shadow_trading_service.shutdown()` ausgeführt (graceful Stop eines laufenden Loops).
4. Typ-Erweiterung (dokumentiert): `snapshot_store: SnapshotStore | PersistedSnapshotStore` in
   `ShadowTradingLoop.__init__` und `ShadowTradingService.__init__` — entspricht der bestehenden
   Union-Typisierung in routes.py; beide Stores haben identische `add()`-Signaturen; kein Cast.

## Dokumentierte Abweichungen vom Workitem-Scope

- Der Dateiscope des Workitems nannte `routes.py`, `main.py`, `tests/test_shadow_trading_api.py`.
  Ergänzt um `services/shadow_trading_service.py`: Business-Logik gehört nicht in HTTP-Routen
  (AGENTS.md); Repo-Muster „ein Service pro Modul".
- Endpunkt-Funktionen heißen `shadow_trading_*` (Kollision mit bestehenden
  `shadow_records`/`shadow_summary` aus `/execution/shadow/*` vermieden).

## Verifikation (echte Ausgaben)

- TDD: Rot-Lauf zuerst (CollectionError: fehlendes Modul), danach 5 dokumentierte Fixrunden:
  1. `decision` ist Konsens-String ("TRADE"/"NO_TRADE"), kein Dict.
  2. `execution_result` ist ein flacher String im Entry (`result["symbols"][0]`); Top-Level enthält
     nur snapshot_id/run_id/decision.
  3. Portfolio-Historie: M2M läuft am Iterationsstart -> Test nutzt 2 Iterationen.
  4. FastAPI inkludiert Router lazy (`_IncludedRouter`) -> Strukturtest liest `router.routes`.
  5. mypy arg-type -> Union-Widening für den Snapshot-Store (siehe oben).
- `uv run pytest tests/test_shadow_trading_api.py -q` -> 13 passed
- `make check` -> 897 passed, 1 warning (bestehende Starlette/httpx-Deprecation);
  ruff: All checks passed!; mypy: Success: no issues found in 56 source files

## Sicherheitsgrenzen (unverändert)

- Die Endpunkte führen Entscheidungen ausschließlich über `ShadowExecutionBackend` -> Paper-Stack aus;
  es gibt keinen Pfad von `/shadow-trading/*` oder dem Lifespan zu Live-Execution.
- Start-Guards: `LIVE_EXECUTION_MUST_BE_DISABLED`, `SHADOW_TRADING_DISABLED`, `NO_SYMBOLS_CONFIGURED`,
  `ALREADY_RUNNING`.
- Kill Switch und Risk Engine bleiben deterministisch im Loop; die API kann keine Policy überschreiben.
