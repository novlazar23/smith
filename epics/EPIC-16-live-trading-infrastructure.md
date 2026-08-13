# EPIC-16: Live Trading Infrastructure — Shadow→Paper→Live Pipeline

## Problem

EPIC-01 bis EPIC-15 haben das komplette Trading-System gebaut: Datenpipeline, Agenten, Konsens, Strategy/Portfolio/Risk, Paper Execution, Governance, Backtesting, Observability und Production Readiness. Das System operiert jedoch ausschließlich in Paper/Shadow-Modus. Der Spezifikation (Abschnitt 9, Analyseauftrag) ist `mode: LIVE` derzeit blockiert durch `LIVE_EXECUTION_BLOCKED = True` in `packages/security/__init__.py` und das Feature-Flag `live_trading_enabled` ist standardmäßig `False`. Abschnitt 32 (Nichtziele MVP) listet "Live Orders" explizit als **nicht im MVP-Umfang**.

EPIC-16 schließt diese Lücke durch eine kontrollierte, phased Transition von Shadow → Paper → Live mit:
- Shadow Mode Trading (echtes Markt-Daten, keine Kapitalrisiken)
- Live Execution Gateway (CCXT-basierte Orderausführung)
- Phasenweise Rollout-Steuerung (konfigurierbare Promotion/Demotion)
- Live Data Pipeline Hardening (Verbindungsgesundheit, Datenqualität)
- Live API Endpunkte (Order-Management, Kill-Switch, PnL)
- Security Hardening (API-Key-Verschlüsselung, RBAC, Audit)

## Ziel

Kontrollierte, auditierbare Transition vom Shadow-Handel zur Live-Execution mit:
- **Shadow Mode**: Echtzeit-Entscheidungen ohne Kapitalrisiko, parallel zu Paper, metrisch vergleichbar
- **Live Execution Gateway**: CCXT-basierte Orderausführung mit Order-State-Machine, Idempotenz, Rate Limiting
- **Phased Rollout**: Konfigurierbare Schwellwerte (Kapital %, Zeit, Brier-Score), automatisierte Promotion/Demotion, Kill-Switch
- **Live Data Pipeline**: Verbindungsgesundheit, Auto-Reconnect, Datenqualitäts-Gates, Failover
- **Live API Endpunkte**: POST/GET Orders, Cancel, Kill-Switch, Health/Readiness, PnL-Tracking
- **Security**: AES-256 Verschlüsselung, Secret Rotation, RBAC, IP-Whitelisting, API-Rate-Limiting

## Abhängigkeiten

- EPIC-02 (Paper Trading Executor, Data Ingestion)
- EPIC-09 (Strategy, Portfolio, Risk, Decision)
- EPIC-10 (Paper Execution — Order Model, Fee Engine, Position Lifecycle)
- EPIC-11 (Governance, Agent Lifecycle, Shadow Mode, Champion-Challenger)
- EPIC-12 (Observability, Testing, Security Review, Go/No-Go Recommendation)
- EPIC-13 (Production Readiness, CI/CD, Hardening)
- EPIC-14 (Backtesting Engine)

## Arbeitspakete

### WP01: Shadow Mode Trading
- `packages/shadow/` — __init__.py, engine.py, metrics.py, comparator.py
- Real-time Decision Engine: Gleicher Orchestrator-Stack wie Production, aber `mode=shadow`
- Parallel Execution: Gleiche Signale an Paper- und Shadow-Pipeline
- Shadow Decision Comparison: Shadow-Entscheidungen vs. Paper-Execution (ohne Kapital)
- Shadow Quality Metrics: Brier Score, Kalibrierung, PnL-Tracking ohne Ausführung
- Latency Logging: Gleiche Latenzmessung wie Production (keine Beschleunigung)
- Audit Trail: Alle Shadow-Entscheidungen protokolliert
- Konfiguration: Shadow-Modus via Feature-Flag und Governance-State-Machine
- File: packages/shadow/*.py, packages/governance/shadow_integration.py

### WP02: Live Execution Gateway
- `packages/live_execution/` — __init__.py, gateway.py, order_state_machine.py, router.py, rate_limiter.py, idempotency.py, validator.py
- CCXT-basierte Order Execution: Unified CCXT API für multi-venue (Binance, etc.)
- Key/Cert Vault Integration: Verschlüsselte API-Keys, Rotation Support
- Order Routing: Single Venue → Multi-Venue (configurable)
- Order State Machine: NEW → PENDING → FILLED / CANCELLED / REJECTED / EXPIRED
  - States: `OrderState.NEW`, `OrderState.PENDING`, `OrderState.PARTIALLY_FILLED`, `OrderState.FILLED`, `OrderState.CANCELLED`, `OrderState.REJECTED`, `OrderState.EXPIRED`, `OrderState.ERROR`
  - Transitions dokumentiert und auditiert
- Rate Limit Management per Venue: Token-basiert, adaptive backoff
- Idempotency Keys: Jeder Order-Submit mit unique idempotency_key
- Order Validation: Vor Submission (size, price, account balance, risk gates)
- Error Handling: Exchange errors, network timeouts, partial fills
- File: packages/live_execution/*.py, packages/execution/venue_client.py ( erweitert)

### WP03: Phased Rollout Controller
- `packages/rollout/` — __init__.py, controller.py, thresholds.py, kill_switch.py, circuit_breaker.py
- Configurable Promotion/Descent:
  - capital_ramp_pct: Schrittweise Kapitalaufstockung (z.B. 1% → 5% → 25% → 50% → 100%)
  - shadow_duration_pct: Mindestzeit in Shadow-Modus (z.B. 30 Tage)
  - min_brier_score: Maximale Brier Score Grenze für Promotion
  - max_drawdown_pct: Maximale Drawdown-Grenze vor Demotion
- Automated Promotion/Demotion:
  - Promotion: Alle Schwellwerte erfüllt + positive Trendlinie + Review
  - Demotion:任一 Schwellwert überschritten → zurück zur vorherigen Phase
- Kill Switch:
  - Manual Kill Switch: API Endpoint + Config-File Override
  - Automatic Kill Switch: Drawdown > threshold, Spread anomaly, Exchange errors > threshold
  - Immediate: Alle Order-Pipeline gestoppt, offene Orders stornoiert
- Circuit Breaker für Exchange Errors:
  - Exchange Error Rate > threshold → Circuit Open
  - Exponential backoff für Wiederherstellung
  - Manual reset erforderlich nach Circuit Open
- File: packages/rollout/*.py, configs/rollout.yaml

### WP04: Live Data Pipeline Hardening
- `packages/live_data/` — __init__.py, health_monitor.py, reconnect.py, quality_gate.py, failover.py, gap_recovery.py
- Connection Health Monitoring:
  - Per-venue health checks (WebSocket, REST, REST backup)
  - Latency tracking, timeout detection
  - Connection pool management
- Auto-Reconnect mit Backoff:
  - Exponential backoff mit jitter
  - Max reconnect attempts konfigurierbar
  - State recovery nach reconnect (orderbook, candle history)
- Data Quality Gates für Live:
  - Freshness: Daten nicht älter als X Sekunden
  - Gap Detection: Fehlende Sequenzen erkennen
  - Price sanity: Unrealistische Preisbewegungen flaggen
  - Volume sanity: Unrealistische Volumen异常
- Failover zu Backup Data Sources:
  - Primary/Backup Venue Konfiguration
  - Automatischer Failover bei Primary-Ausfall
  - Data consistency zwischen Primary und Backup
- Data Gap Recovery:
  - Gap Detection → Gap Report
  - Historical data replay zum Schließen von Lücken
  - Orderbook reset nach Gap > threshold
- File: packages/live_data/*.py, packages/ingestion/*.py ( erweitert)

### WP05: Live API Endpoints
- `apps/api/routers/live_orders.py` — Live Order Management Endpunkte
- POST /v1/live/orders — Live Orders submit mit Audit Trail
  - Request: instrument, direction, quantity, order_type, price (optional), idempotency_key, venue
  - Response: order_id, state, submitted_at, idempotency_key
  - Validation: All orders validated before submission
  - Audit: Alle Orders im Audit Trail geloggt
- GET /v1/live/orders — Order History
  - Query: order_id (optional), status (optional), venue (optional), from (datetime), to (datetime)
  - Response: List of orders with state, fills, timestamps
- POST /v1/live/cancel — Order Cancel
  - Request: order_id, reason (optional)
  - Response: Cancel request status, order state transition
- POST /v1/live/kill-switch — Kill Switch Control
  - Request: action (activate/deactivate), reason (mandatory)
  - Response: Kill switch state, affected orders, confirmation
- Health/Readiness für Live Mode:
  - GET /v1/health/live — Live-mode spezifischer Health Check
  - Liveness: Prozess läuft
  - Readiness: Connection zu Exchange, Feature-Flag enabled, Rollout phase > SHADOW
- Live PnL Tracking:
  - GET /v1/live/pnl — Realized/Unrealized PnL
  - GET /v1/live/pnl/daily — Tägliche PnL Aggregation
  - Metrics: Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor
- File: apps/api/routers/live_orders.py, apps/api/routers/live_health.py

### WP06: Security Hardening
- `packages/security/hardening/` — encryption.py, secret_rotation.py, audit_live.py, rbac_live.py, ip_whitelist.py, api_rate_limiter.py
- API Key Encryption at Rest (AES-256):
  - AES-256-GCM Verschlüsselung für API-Keys im Storage
  - Master Key aus ENV/Vault (nicht im Code)
  - Key Rotation unterstützt
- Secret Rotation Support:
  - Configurable rotation interval (z.B. 90 Tage)
  - Zero-downtime rotation: New key parallel, Old key deprecate
  - Rotation audit trail
- Audit Trail für alle Live Operations:
  - Jeder Order Submit, Cancel, Fill im Audit Trail
  - Kill Switch Aktionen auditet
  - Rollout State Changes auditet
  - Unveränderbares Audit Log (append-only)
- Role-Based Access Control (RBAC) für Live Operations:
  - execute_live: Required für POST /v1/live/orders
  - cancel_orders: Required für POST /v1/live/cancel
  - manage_kill_switch: Required für POST /v1/live/kill-switch
  - view_live_pnl: Required für GET /v1/live/pnl*
  - Rollen aus packages/security/__init__.py erweitert
- IP Whitelisting:
  - Konfigurierbare Whitelist für Live API Endpunkte
  - Blockiert nicht-whitelisted IPs
  - Audit: Failed IP attempts logged
- API Rate Limiting:
  - Per-IP rate limiting für Live Endpunkte
  - Configurable limits (z.B. 10 orders/min, 100 requests/min)
  - Rate limit headers in responses
- File: packages/security/hardening/*.py

## DoD

- Shadow Mode: Parallel zu Paper, Brier Score wird getrackt, keine Kapitalrisiken
- Live Gateway: CCXT-basiert, Order State Machine vollständig, Idempotenz garantiert
- Rollout Controller: Konfigurierbare Schwellwerte, automatisierte Promotion/Demotion
- Kill Switch: Manual + Automatic (Drawdown/Spread/Exchange Errors), sofortige Wirkung
- Data Pipeline: Health Monitoring, Auto-Reconnect, Gap Detection, Failover
- API Endpunkte: Alle 6 Endpunkte implementiert, dokumentiert, getestet
- Security: AES-256 Verschlüsselung, Secret Rotation, RBAC, IP Whitelist, Rate Limiting
- Alle Tests bestanden, ruff clean
- Go/No-Go Recommendation von EPIC-12 WP05 berücksichtigt

## Risiken

- CCXT API Breaking Changes zwischen Exchange Updates
- Rate Limits der Exchanges unerwartet niedrig
- Kill Switch False Positives → unnötige Stoppage
- Data Gaps während Failover → inkonsistente State
- Security: API Key Leak → sofortiger Kompromiss
- Latenzunterschied Shadow vs. Production

## Rollback

- Kill Switch aktiviert → alle Order-Pipeline sofort gestoppt
- Rollback: Shadow Mode zurück, Feature-Flag deaktiviert
- Database State kann zurückgesetzt werden (Order History unveränderbar)
- Secret Rotation: Alte Keys in Deprecation Window akzeptiert

## Definition of Done (Specification Section 32 Ergänzung)

Neue Kriterien für Live Phase:
1. Shadow Mode läuft parallel zu Paper ohne Kapitalrisiko
2. Brier Score und Kalibrierung in Shadow gemessen
3. Live Order State Machine: Alle 8 States implementiert
4. Order State Transitions auditiert und unveränderbar
5. CCXT Unified API für Order Execution verwendet
6. Idempotency Keys: Duplicate Submits rejected
7. Rate Limiting per Venue implementiert
8. Kill Switch (manual + automatic) funktionstüchtig
9. Data Pipeline: Health Monitoring + Auto-Reconnect aktiv
10. API Endpunkte: Alle 6 Endpunkte mit dokumentierten Request/Response
11. AES-256 Encryption für API Keys at rest
12. RBAC: execute_live permission required für Live Orders
13. IP Whitelisting konfiguriert und aktiv
14. API Rate Limiting pro IP implementiert
15. Audit Trail: Alle Live Operations unveränderbar geloggt
16. Secret Rotation getestet und dokumentiert
