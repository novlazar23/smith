# Phase 4 — Paper Trading Epic

## Dependencies
- Phase 1 (Research Runtime) — ✅ COMPLETE
- Phase 2 (Evaluation) — ✅ COMPLETE
- Phase 3 (Evolution) — ✅ COMPLETE
- Spec: `specs/phase-4-paper-trading.md`

## Work Items

### WI-P4-1: PaperExchange Core
**Ziel**: PaperExchange-Klasse mit Slippage, Fill Rate, Fee-Modell
**Eingaben**: spec `specs/phase-4-paper-trading.md`, R4.1-R4.4
**Dateien**:
- `src/trading_harness/services/paper_exchange.py` — new
- `tests/test_paper_exchange.py` — new
**Akzeptanz**:
- Slippage berechnet korrekt basierend auf slippage_bps
- Fill Rate angewendet (0.8 ± 5% jitter)
- Fees berechnet (0.1% per trade)
- Order Status transitions: PENDING → FILLED/PARTIALLY_FILLED/REJECTED
- 15+ Tests, alle grün

### WI-P4-2: PaperPosition & PositionManager
**Ziel**: Position lifecycle management mit Stop-Loss/Target
**Eingaben**: WI-P4-1 (PaperExchange), spec `specs/phase-4-paper-trading.md`, R4.5-R4.8
**Dateien**:
- `src/trading_harness/services/position_manager.py` — new
- `tests/test_position_manager.py` — new
**Akzeptanz**:
- PaperPosition erstellt bei FILLED
- Partial Close unterstützt
- Stop-Loss/Take-Profit automatisch ausgelöst
- PnL berechnet korrekt (LONG/SHORT)
- 15+ Tests, alle grün

### WI-P4-3: PortfolioTracker
**Ziel**: Aggregierte Portfolio-Metriken tracken
**Eingaben**: WI-P4-2 (PositionManager), spec `specs/phase-4-paper-trading.md`, R4.9-R4.10
**Dateien**:
- `src/trading_harness/services/portfolio_tracker.py` — new
- `tests/test_portfolio_tracker.py` — new
**Akzeptanz**:
- Equity Curve korrekt (start + unrealized + realized PnL)
- Drawdown berechnet (max, current)
- Pro-Symbol exposure tracking
- 10+ Tests, alle grün

### WI-P4-4: PaperExecutionService
**Ziel**: Integration von PaperExchange + PositionManager + PortfolioTracker
**Eingaben**: WI-P4-1, WI-P4-2, WI-P4-3, spec `specs/phase-4-paper-trading.md`, R4.17
**Dateien**:
- `src/trading_harness/services/paper_execution_service.py` — new
- `tests/test_paper_execution_service.py` — new
**Akzeptanz**:
- Full flow: TradeProposal → PaperTrade → PaperPosition → PnL
- Integration mit TradingRun-Flow
- 10+ Tests, alle grün

### WI-P4-5: Datenbank-Persistenz
**Ziel**: PostgreSQL-backed stores für PaperTrading mit In-Memory-Fallback
**Eingaben**: WI-P4-1 bis WI-P4-4, spec `specs/phase-4-paper-trading.md`, R4.13-R4.15
**Dateien**:
- `src/trading_harness/services/db.py` — extend INIT_SQL
- `src/trading_harness/services/paper_trade_store.py` — new (Persisted)
- `src/trading_harness/services/paper_position_store.py` — new (Persisted)
- `src/trading_harness/services/portfolio_store.py` — new (Persisted)
- `tests/test_paper_trade_store.py` — new
- `tests/test_paper_position_store.py` — new
- `tests/test_portfolio_store.py` — new
**Akzeptanz**:
- paper_trades, paper_positions, portfolio_state Tabellen
- In-Memory-Fallback wenn PostgreSQL nicht verfügbar
- 20+ Tests, alle grün

### WI-P4-6: API Routes
**Ziel**: Alle Paper Trading Endpunkte in routes.py
**Eingaben**: WI-P4-1 bis WI-P4-5, spec `specs/phase-4-paper-trading.md`, R4.16
**Dateien**:
- `src/trading_harness/api/routes.py` — extend
- `tests/test_api_paper.py` — new
**Akzeptanz**:
- Alle 12 Endpunkte implementiert (R4.16)
- POST /paper/run/{id}/execute integriert in TradingRun-Flow
- 20+ Tests, alle grün

### WI-P4-7: Dokumentation & Cleanup
**Ziel**: handoff.md aktualisieren, docs aktualisieren
**Eingaben**: Alle vorherigen WIs
**Dateien**:
- `docs/handoff.md` — update
- `README.md` — update Phase 4 status
**Akzeptanz**:
- handoff.md: Phase 4 als COMPLETE markiert
- Alle Abhängigkeiten dokumentiert
- make check clean

## Critical Path
WI-P4-1 → WI-P4-2 → WI-P4-3 → WI-P4-4 → WI-P4-5 → WI-P4-6 → WI-P4-7

## Parallelisierung
- WI-P4-1, WI-P4-2, WI-P4-3 können parallel starten (unabhängig)
- WI-P4-4 benötigt WI-P4-1, WI-P4-2, WI-P4-3
- WI-P4-5 benötigt WI-P4-1 bis WI-P4-4
- WI-P4-6 benötigt WI-P4-5
- WI-P4-7 kann parallel zu WI-P4-6 starten (dokumentation-only)

## Risiken
- R1: PaperExchange muss deterministic sein (kein random ohne seed)
- R2: Position sizing calculations müssen korrekt sein (leverage × equity / stop_distance)
- R3: Datenbank-Migrationen müssen versioniert sein
- R4: Keine Änderungen an bestehender ExecutionGateway-Logik

## Definition of Done
- Alle WIs abgeschlossen
- make check clean (tests, lint, mypy)
- docs/handoff.md aktualisiert
- Spec artifact created
