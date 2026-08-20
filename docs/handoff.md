# Development Handoff

## Current state

No implementation task is currently in progress. The repository provides a reproducible Python 3.12
environment, a frozen dependency lock, matching local/CI checks, and versioned OpenCode rules and
commands. The complete German OpenCode usage and cross-device workflow is documented in
`docs/opencode-nutzung.md`.

**Harness-Run CLOSEOUT (2026-08-20):** Der Harness-Run `RUN-20260818-115553` ist
`completed`. Alle 22 Arbeitspakete (WI-P4-1…WI-P4-7, WI-P5-1…WI-P5-15) sind `completed`
und unabhängig reviewed — Gate-Check: 22/22 latest Reviews `approved` (26 Reviews gesamt,
Review-IDs 1–26). „Wave 2" (2026-08-20, 4 unabhängige Reviewer) plus Re-Reviews haben die
letzten offenen Pakete geschlossen:

- WI-P5-8 (Review 18, changes-requested): fehlende `docs/security-review-phase5.md` —
  erstellt (Commit `64159df`, 14 Kontrolle-Sektionen + nicht blockierende Befunde:
  `max_leverage 2.0` in `config/risk-policy.yaml:12` vs. README „1.0x", fail-open API-Auth
  ohne konfigurierte Keys, unauthentizierte Legacy-Endpunkte); Re-Review 25 (unabhängig):
  **approved**.
- WI-P4-4 (Review 21, changes-requested): ausgelieferte Paper-Wiring in `routes.py` baute
  `PaperExchange()` ohne Stores — jede Paper-Order → `RuntimeError('PaperExchange stores
  not configured')` (3x in Folge → Kill-Switch-Auto-Trigger); PositionManager/PortfolioTracker
  nirgends in src/ verdrahtet. TDD-Fix (Commit `f89a742`): neue Factory
  `src/trading_harness/services/paper_execution_stack.py` — PaperExchange mit
  Persisted*Stores, PositionManager, PortfolioTracker; Fill-Flow TradeProposal → PaperTrade →
  PaperPosition → PortfolioState/PnL; `routes.py` baut den Stack via
  `build_paper_execution_stack(db=_db)`; `PaperExchangeAdapter`-Default jetzt sicher
  (In-Memory-Store) + `on_fill`-Callback; 11 neue Wiring-Tests (vor dem Fix rot, nach dem
  Fix grün — Stash-Experiment belegt beide Zustände). Re-Review 26 (R8, unabhängig):
  **approved** — alle 3 Befunde mit Zitation verifiziert, Sicherheitsinvarianten intakt.
- Verifikation (2026-08-20, Orchestrator, HEAD `f89a742`): `make check` — 784 Tests grün,
  ruff clean, mypy clean (51 source files); `data/kill_switch.json` byte-identisch
  (sha256 `1389df52f9a2e125a05a2ee96b13263870a234236506d2487c12cbc06d2383a9`).
  Live Execution bleibt standardmäßig deaktiviert; keine Sicherheitsgrenze wurde geändert.

**Harness-Buchhaltung (abgeschlossen 2026-08-20):** Die 7 Phase-4-Pakete
(WI-P4-1…WI-P4-7) wurden rückwirkend `completed` geschlossen, mit strukturierten,
evidenzbasierten Results (Arbeit in der damaligen Session umgesetzt und verifiziert;
Commits 77f0d37, 93ca0b4, 65e4d17, 6a1f202; dokumentierte Abweichungen: Pipeline-Wiring
statt `paper_execution_service.py`, SQLite statt PostgreSQL, keine 12 dedizierten
Endpoints). Der Übergang `execution-running → completed` wurde am 2026-08-20 nach dem
22/22-`approved`-Gate ausgeführt (siehe Closeout oben).

**Review-9-Fix (WI-P5-2 RateLimiter, Commit `e109400`):** Die Refill-Rate skaliert jetzt
mit dem Limit (`limit/60` Tokens/s pro Bucket) — Sustained-Rate = konfiguriertes Limit
(10/min global, 2/min pro Symbol), Burst-Kapazität = Limit unverändert;
`LiveExecutionService` baut den RateLimiter aus
`ExecutionConfig.global_rate_limit`/`symbol_rate_limit` (Defaults 10/2; war vorher
hardcodiert und wirkte auch nicht über `routes.py`). +6 Tests in `test_rate_limiter.py`
(14 gesamt), +2 Regressionstests in `test_live_execution_service.py`.

**Review-13-Fix (WI-P5-6 ExecutionLogStore, Commit `4e4cd38`):** Entry-IDs werden jetzt
vollständig im Lock erzeugt (Race beseitigt); +4 Tests in `test_execution_store.py`
(11 gesamt: `clear()`, korrumpiertes State-File, atomare Write-Crash-Integrität,
Concurrency); `docs/phase5-epic.md` (WI-P5-6) enthält die dokumentierte, datierte
(2026-08-20) Architektur-Entscheidung: JSON-Datei-Persistenz statt `execution_log`-
Tabelle in `db.py` (redundanter zweiter Persistenzmechanismus; JSON ist die genehmigte
Architektur laut WI-P5-15/Review 6).

**Verifikation (2026-08-20, Orchestrator, HEAD `e109400`):** `make check` — 772 Tests
grün, ruff clean, mypy clean (50 source files); `data/kill_switch.json` byte-identisch
(sha256 `1389df52f9a2e125a05a2ee96b13263870a234236506d2487c12cbc06d2383a9`).

**Offene NITs aus Review-ID 6/7 geschlossen (akzeptiert, won't-fix):** die fehlenden
End-zeilenumbrüche in `execution_store.py`/`test_api_execution.py`/`test_kill_switch.py`
sind ein Repository-weites Muster — 67 Dateien in `src/`, `tests/`, `docs/`, `config/`
ohne End-zeilenumbruch (ruff-konform, W292 inaktiv, kein funktionaler Effekt). Eine
partielle "Reparatur" würde Inkonsistenz erzeugen; ein repository-weiter 1-Byte-Diff
wäre reiner Lärm ohne Mehrwert.

**Nächste Schritte (2026-08-20, mit User freigegeben):**
1. Nächstes Meilenstein-Epic laut README Abschnitt 8: „Die erste produktive
   Ausbaustufe soll ausschließlich Shadow Trading durchführen" — Shadow-Trading-Epic
   wird als nächster Harness-Run angestoßen (problem → spec → epic).
2. Nicht blockierende Befunde aus `docs/security-review-phase5.md` (max_leverage-Drift
   2.0 vs. 1.0x, fail-open API-Auth, unauthentizierte Legacy-Endpunkte) — Kandidaten
   für das Shadow-Trading-Epic bzw. ein separates Security-Workitem.
3. Push von `main` auf `origin/main` mit expliziter User-Freigabe beauftragt; erfolgt
   direkt nach diesem Closeout-Commit.

Phase 1 (Research Runtime) additions committed:
- `TradingRun` model with full lifecycle state machine (`RunState`)
- `PerformanceRecord` and `AuditEntry` models
- `TradingRunService` — thread-safe run lifecycle manager with audit trail
- `PerformanceStore` — thread-safe performance records store
- API endpoints: `/runs`, `/runs/{id}`, `/runs/{id}/transition/{state}`, `/runs/{id}/decision`,
  `/runs/{id}/complete`, `/runs/{id}/fail`, `/performance`, `/performance/summary/run/{id}`,
  `/performance/summary/agent/{id}`, `/audit`, `/audit/{entity_id}`
- Policy loader resolves Docker paths (`/app/config/...`) to local paths automatically

Phase 2 (Evaluation) additions committed:
- `MarketRegime` enum — 9 market regimes (strong_bull, weak_bull, range, weak_bear, strong_bear,
  high_volatility, low_volatility, crash, recovery)
- `OutcomeRecord` — actual market outcome for prediction evaluation with MFE/MAE
- `EvaluationResult` — evaluation result storage with metric name and value
- `WalkForwardResult` — per-window walk-forward evaluation result
- `OutcomeGenerator` — generates outcome records from predictions + actual market data,
  stores/retrieves outcomes by agent/run/regime
- `EvaluationService` — comprehensive evaluation metrics engine:
  - Brier Score (probabilistic prediction quality)
  - Expected Calibration Error (ECE) with 5-bin grouping
  - Expectancy (weighted average return per trade)
  - MFE/MAE statistics (avg/max favorable/adverse excursion)
  - Confusion matrix (TP/FP/TN/FN), Precision, Recall, F1
  - Directional Accuracy
  - Per-regime performance evaluation
  - Drawdown calculation (max drawdown, current drawdown, recovery periods)
  - Out-of-Sample evaluation with degradation ratio
  - Walk-Forward stability evaluation with rolling windows
- API endpoints: `/outcomes`, `/outcomes/agent/{id}`, `/outcomes/run/{id}`, `/outcomes/regime/{id}`,
  `/evaluation/agent/{id}`, `/evaluation/regime/{id}/{regime}`,
  `/evaluation/drawdown/{id}`, `/evaluation/out-of-sample`, `/evaluation/walk-forward/{id}`,
  `/evaluation/results`, `/evaluation/results/agent/{id}`

Phase 1 Persistence (PostgreSQL-backed stores) additions committed:
- `Database` (`db.py`) — async PostgreSQL connection pool with schema migration (agents, market_snapshots,
  audit_log, trading_runs, outcomes, evaluation_results, performance_records), graceful fallback to in-memory on connection failure
- `PersistedAgentRegistry` — PostgreSQL-backed agent registry with in-memory fallback; supports add/list/get/version
- `PersistedSnapshotStore` — PostgreSQL-backed market snapshot store with content-hash (SHA-256) integrity;
  in-memory fallback when DB unavailable
- `PersistedPerformanceStore` — PostgreSQL-backed performance records store with in-memory fallback;
  supports add/get/all/by_run/by_agent/by_snapshot and upsert-on-conflict
- All stores use `is_available` guard so they never silently swallow errors or block on failed connections
- `datetime` fields correctly parse to `datetime` objects (not strings) from Postgres
- API routes wired to persistent stores; fallback mode works without Postgres running
- Performance store test suite: 11 tests covering fallback add/get/all filter methods, upsert/overwrite, and defaults
- `GET /performance/summary/snapshot/{snapshot_id}` endpoint added to routes

Phase 4 — Paper Trading: ✅ COMPLETE. 12 Dateien, 2840 Zeilen, 323 Tests grün.

Phase 5 — Live Execution: ✅ CORE SERVICES + Read/Trade API Auth + Crypto Adapters + Shadow Mode + Network Isolation + Credential Management + Minimal Capital + Safety Gate COMPLETE.
11 Services, 15 Testdateien, 534 Tests grün.

Services implementiert:
- `KillSwitch` — thread-safe, SQLite-persistiert, 11 Tests
- `RateLimiter` — Token Bucket, global + pro Symbol, RLock, 8 Tests
- `OrderDeduplicator` — memory-bounded deque, periodischer Trim, 11 Tests
- `ExchangeAdapter` — abstrakte Schnittstelle + StubExchangeAdapter, 8 Tests
- `PaperExchangeAdapter` — bridge PaperExchange→ExchangeAdapter (submit_order, cancel/
  get_status/balance/ticker stubs), 27 Tests
- `LiveExecutionService` — orchestrates KillSwitch→RateLimiter→Deduplicator→Exchange, 22 Tests
- `ExecutionLogStore` — JSON-Persistenz, in-memory Fallback, 7 Tests
- API Routes: `POST /execution/orders`, `POST /execution/kill-switch/{enabled}`, `GET /execution/status`, `GET /execution/logs` — 15 Tests
- `API security` — `require_trade_key` / `require_read_key` Dependencies, Header+Query-Parsing, 16 Tests

Read/Trade API Separation (R5.21–R5.22):
- `POST /execution/orders` — erfordert `X-Trade-API-Key` (config: `trade_api_key`)
- `POST /execution/kill-switch/{enabled}` — erfordert `X-Trade-API-Key`
- `GET /execution/status` — erfordert `X-Read-API-Key` (config: `read_api_key`)
- `GET /execution/logs` — erfordert `X-Read-API-Key`
- Key-Parsing: Header (`X-Trade-API-Key`/`X-Read-API-Key`) oder Query-Param (`trade_api_key`/`read_api_key`)
- Backward-compatible: Wenn Key nicht konfiguriert, werden Endpoints durchgelassen

Sicherheitsgrenzen:
- Live Execution: standardmäßig `default: false`
- Kill Switch: standardmäßig `default: true` (aktiviert)
- Keine echte Exchange-Integration im MVP
- Paper-Adapter ersetzt nur `StubExchangeAdapter` — alle Sicherheitsgrenzen unverändert

Phase 5 — Crypto Exchange Adapters + Shadow Mode: ✅ COMPLETE.
5 neue Dateien, 2 Testdateien, 34 neue Tests (575 gesamt grün).

Crypto Exchange Adapters:
- `BaseCryptoExchangeAdapter` — abstrakte Basisklasse mit shared Signatur-Generierung,
  HMAC-SHA256, httpx Client, `simulated=True` sicherer Standard, optionaler `passphrase`-Parameter
  für Bitget/Bybit ACCESS-PASSPHRASE, `_sign_request` gibt `dict[str, str] | str` (adapter-spezifisch)
- `BybitExchangeAdapter` — Bybit V5 API: `/v5/order/create`,
  Signatur via `X-BAPI-SIGN` Header (HMAC-SHA256 hex), `X-BAPI-TIMESTAMP`, `X-BAPI-API-KEY`,
  `X-BAPI-RECV-WINDOW`. Signing: `timestamp + apiKey + recvWindow + jsonBody` (POST)
- `BitgetExchangeAdapter` — Bitget V3 (UTA) API: `/api/v3/trade/place-order`,
  HMAC-SHA256 via `ACCESS-SIGN` Header (base64-encoded), `ACCESS-TIMESTAMP`, `ACCESS-KEY`,
  `ACCESS-PASSPHRASE` (identity-only, not used for HMAC). Signing: `timestamp + POST + path + body`
- `BinanceExchangeAdapter` — Binance V4 Spot API: `/api/v4/trade/order`,
  HMAC-SHA256 als query-param `signature`, `X-MBX-APIKEY`, `X-MBX-TIME`.
  Override von `_make_signed_request` (signature in params, nicht headers),
  flat response parsing (`{code, msg, orderId}` statt `result`-Wrapper)
- `CoinbaseExchangeAdapter` — Coinbase Advanced Trade API: `/api/v3/brokerage/orders`,
  HMAC-SHA256 base64-encoded von `timestamp + METHOD + requestPath + body`,
  `CB-ACCESS-SIGN`, `CB-ACCESS-TIMESTAMP`, `CB-ACCESS-KEY`, `CB-ACCESS-PASSPHRASE`
- Alle Adapter: `submit_order()` → returns `order_id, status, raw`
- Alle Adapter: simulieren Fills wenn keine Credentials konfiguriert (`simulated=True`)
- Keine echten Orders ohne explizite API-Schlüssel in `.env`
- Response-Validierung: `_validate_response` prüft `retCode == "0"` (Bybit) und `code == "0"` (Bitget/Binance)
- Rate-Limit-Handling: HTTP 429 → Retry mit exponentiellem Backoff (1s, 2s, 4s)
- Transiente Fehler: HTTP 5xx, httpx.TimeoutException, httpx.ConnectError → Retry mit Backoff (0.5s, 1s, 2s)
- Auth-Fehler (401/403): kein Retry, sofortiger Fehler

Signatur-Fix (853387b): Korrigierte HMAC-SHA256-Signierung für beide Exchanges:
- Bybit: war query string → jetzt `timestamp + apiKey + recvWindow + jsonBody`
- Bitget: war hardcoded `GET /api/v3/spot/trade/order` mit hex-output → jetzt
  `timestamp + POST + /api/v3/trade/place-order + body` mit base64-output
- `_simulate`: fragile URL-Matching bereinigt (removierte `/spot/trade/order` die Cancel-URL trafen)

Shadow Mode Logging:
- `ShadowModeLogger` — speichert alle Shadow-Orders als `ShadowModeRecord` Pydantic Models
  in einer in-Memory `deque` mit optionalem `maxlen`
- `ShadowModeAdapter` — `ExchangeAdapter`-Konform, ruft intern `submit_order()` auf Logger
  statt echter Exchange-API, loggt Fill-Preis mit konfigurierbarer Slippage
- API Endpunkte:
  - `POST /execution/shadow/submit` — Order in Shadow-Mode loggen (erfordert Trade-Key)
  - `GET /execution/shadow/summary` — Zusammenfassung aller Shadow-Orders (erfordert Read-Key)
  - `GET /execution/shadow/records` — Records abrufbar, optional gefiltert nach
    `decision_id`, `symbol`, `run_id` (erfordert Read-Key)
- Crypto Status Endpoint: `GET /execution/crypto/status` — zeigt welche Adapter
  simuliert vs. live konfiguriert sind

Pipeline-Integration (Phase 6): Crypto-Adapter durch volle Execution-Pipeline.
- `CryptoExecutionRouter` — Singleton-Router der Orders an den korrekten Exchange-Adapter routet
  (bybit, bitget, binance, coinbase). Factory `_build_adapter()` instanziiert lazy pro exchange_name.
- `crypto_execution_service` — eigenständiger `LiveExecutionService` mit `_crypto_router` als Adapter,
  nutzt denselben `execution_kill_switch`/`credential_manager`/`network_policy` wie Paper-Service.
- Pipeline für Crypto-Orders identisch: KillSwitch → RateLimit → Dedup → SymbolWhitelist → RiskEngine
  → NetworkPolicy → CredentialCheck → Exchange → Log.
- `POST /execution/crypto/submit` — unified Endpoint statt separater `/bybit`/`/bitget`/`/binance`/`/coinbase`
  Routen. Payload: `decision_id`, `run_id`, `symbol`, `side`, `quantity`, `price`.
- `_get_exchange_url()` in `LiveExecutionService` erweitert: Binance (`https://api.binance.com/*`),
  Coinbase (`https://api.coinbase.com/*`), Router (`https://*`).
- API Routen `/execution/crypto/bybit`, `/bitget`, `/binance`, `/coinbase` entfernt —
  ersetzt durch `POST /execution/crypto/submit`.
- `execution_status` zeigt nun `live_execution_enabled` über `live_execution_service.is_live_enabled`.

Neue API Endpunkte in `routes.py`:
- `POST /execution/crypto/submit` — unified Crypto-Submit durch LiveExecutionService-Pipeline
- `GET /execution/crypto/status` — Crypto-Router Status (simulated=True/False, supported_exchanges)

Dynamic Credential Loading (Phase 7): CryptoExecutionRouter prüft Credentials pro Exchange.
- `CryptoExecutionRouter.CREDENTIAL_PREFIXES` — Mapping: exchange → (KEY_ENV, SECRET_ENV)
  `{"bybit": ("BYBIT_API_KEY", "BYBIT_API_SECRET"), "bitget": (...), "binance": ..., "coinbase": ...}`
- `_resolve_adapter_state(exchange)` — liest Credentials via `CredentialManager.get()`,
  gibt `(simulated: bool, kwargs: dict)` zurück
- Existieren BEIDE Credentials (KEY + SECRET) → `simulated=False` → echte API-Orders
- Fehlendes Credential → `simulated=True` → simulierte Orders (sicherer Fallback)
- State wird gecached pro Exchange (`self._adapter_state`) — keine wiederholten Lookups
- `crypto_execution_service` übergibt `credential_manager=credential_manager` an Router
- `crypto_status()` zeigt `credential_states` dict: `{exchange: "LIVE" | "SIMULATED"}`
- `router.close()` cleared `_REGISTRY` und `_adapter_state`

Phase 8 — Order Status/Cancel Delegation via Exchange Name: ✅ COMPLETE.
- `LiveExecutionService.submit_order()` akzeptiert jetzt `exchange_name: str | None` Parameter,
  leitet an Router/Adapter weiter
- `LiveExecutionService.get_order_status()` — Pipeline-geschützt (KillSwitch, NetworkPolicy),
  delegiert an `ExchangeAdapter.get_order_status(order_id, exchange_name)`
- `LiveExecutionService.cancel_order()` — Pipeline-geschützt, delegiert an
  `ExchangeAdapter.cancel_order(order_id, exchange_name)`
- Neue API Endpunkte:
  - `GET /execution/crypto/status/{order_id}` — Order-Status via Pipeline (Read-Key)
  - `DELETE /execution/crypto/cancel/{order_id}` — Order-Cancel via Pipeline (Trade-Key)
  - `POST /execution/crypto/submit` Payload: optionaler `exchange_name` Feld
- Alle Adapter-Signaturen erweitert: `submit_order`, `get_order_status`, `cancel_order`
  akzeptieren jetzt `exchange_name: str | None = None`
- `ExchangeAdapterError` als spezifischer Exception-Handler in `LiveExecutionService`
- `LiveExecutionService._get_exchange_url()` — erweitert für `CryptoExecutionRouter`
  (`https://api.*/*` Pattern für Network Policy)
- 8 neue Tests: router status/cancel/ticker simulation, API crypto submit/status/cancel routes
- 592 Tests grün, ruff + mypy clean

Phase 9 — HTTP-Level Adapter Validation Tests: ✅ COMPLETE.
39 neue Tests (433 Zeilen), 628 Tests gesamt grün, ruff + mypy clean.

HTTP-Level Request Construction:
- Bybit: 7 Tests — submit_order/get_order/cancel/balance/ticker URLs, Signature String Format, Headers
- Bitget: 7 Tests — submit_order/get_order/cancel/balance/ticker URLs, HMAC base64-Signierung, Headers
- Binance: 5 Tests — submit_order/ticker/balance URLs, query-param signature, X-MBX-APIKEY header
- Coinbase: 4 Tests — submit_order/ticker/balance URLs, CB-ACCESS-KEY headers
- Gesamte URL-Korrektheit aller Adapter-Methoden verifiziert ohne echte Credentials

Retry Behavior:
- RateLimitError (HTTP 429): retry 3x dann Raise, call_count verifiziert
- ConnectionError (HTTP 5xx): retry 3x dann Raise, backoff verifiziert
- Retry-then-Success: 429 → 200 funktioniert korrekt, call_count == 2
- Auth-Failure (HTTP 401): sofortiger Fehler, kein Retry, call_count == 1

Response Parsing — Exchange-spezifisch:
- Bybit: order_id aus response, walletBalance[0].totalBalance parsing, ticker bidPx/askPx/lastPx
- Bitget: data-wrapper parsing, balance response format
- Binance: flat response `{code, msg, orderId}`, invalid code → ResponseValidationError
- Coinbase: order_id aus response, accounts array parsing, USDT symbol handling fixed (fallback war "BTC" → jetzt symbol)

Bugfix: Coinbase `get_balance` base_currency fallback — vorher `or "BTC"` (fand BTC-Balance 1.5 für USDT-Request), jetzt `or symbol` (korrekte USDT-Balance 100000).

Mypy fixes (pre-existing):
- `order_deduplicator.py:42` — `maxlen` kann `None` sein (deque ohne maxlen)
- `portfolio_tracker.py:13,17,19` — Protocol-Methoden mit `...` als Body
- `portfolio_tracker.py:89` — `Database` undefined → `Any` Type
- `portfolio_tracker.py:199,200` — type annotation auf `p` in generator expression
- `paper_exchange.py:59` — `stores` kann `None` sein → Runtime-Check hinzugefügt

BLE001 Lint fixes (pre-existing):
- `live_execution_service.py:164` — `except Exception` mit `# noqa: BLE001`
- `test_kill_switch.py:92` — concurrent stress test
- `test_order_deduplicator.py:79` — concurrent stress test
- `test_paper_exchange.py:462,491,499` — concurrent stress tests

Phase 10 — Connection/Error-Handling & Edge-Case Tests: ✅ COMPLETE.
22 neue Tests (347 Zeilen), 645 Tests gesamt grün, ruff + mypy clean.

ConnectionError-Retry (simulated=False):
- DNS-Fehler: 3x retry dann ConnectionError mit message
- Timeout: 3x retry dann ConnectionError
- SSL-Fehler: 3x retry dann ConnectionError
- Alle 4 Adapter (Bybit, Bitget, Binance, Coinbase) abgedeckt

HTTP-400 Response Handling:
- Bybit 400: retCode "-1017" → ExchangeAdapterError mit Error-Code
- Binance 400: code -1121 → ExchangeAdapterError
- Coinbase 400: error "invalid_order" → ExchangeAdapterError

Response-Schema Edge Cases:
- Empty ticker lists, missing fields, null values
- Default-Werte bei fehlenden bidPx/askPx/lastPx
- Balance ohne walletBalance/accounts → graceful handling
- Order-Status ohne data-Objekt

CryptoExecutionRouter Live-Mode:
- Credential-Loading: beide Keys nötig für simulated=False
- Mock-basierter Test: resolve_adapter_state prüft KEY+SECRET

Bugfix in dieser Phase:
- Binance `_make_signed_request`: ConnectError now raises ConnectionError with original message

## Next priority

Phase 5 — Live Execution: ✅ **KOMPLETT MIT RISK ENGINE HARDENING**. Alle Core Services,
Read/Trade API Auth, Crypto Adapters, Shadow Mode, Network Isolation (R5.15–R5.17),
Credential Management (R5.18–R5.20), Risk Engine Hardening (R5.2–R5.4) abgeschlossen.

Full Execution Pipeline (LiveExecutionService):
  KillSwitch → RateLimit → Dedup → SymbolWhitelist → RiskEngine → NetworkPolicy →
  CredentialCheck → Exchange → Log

Risk Engine Hardening (R5.2–R5.4):
- `ExecutionConfig.symbol_whitelist` — Symbols müssen in Whitelist sein (R5.6)
- `ExecutionConfig.allowed_exchanges` — erlaubte Exchanges konfigurierbar
- `RiskEngine.evaluate()` wird vor jedem Submit aufgerufen
- Position Sizing: `RiskDecision.max_position_size` wird enforced
- Daily Loss Limit, Leverage Limit, Risk/Rewards werden validiert
- Risk-Log in `ExecutionLog.risk_approved` und `risk_max_position_size`

Nächste Schritte (in Reihenfolge):

1. **Exchange-Adapter Live-Integration** — echte API-Orders für Bybit/Bitget/Binance/Coinbase
   - Response-Validierung gegen Exchange-Schemas ✅ DONE: Pydantic Models für alle 4 Adapter (Bybit V5, Bitget V3, Binance V4, Coinbase Pro); `_validate_response` mit ExchangeResponseError + 3-Fallback-Fallback (msg→retMsg→message); 35 Tests (132 adapter tests total, 680 project total)
   - Rate-Limit-Error-Handling (429, timeout retries) ✅ DONE: Phase 10 — HTTP 429 retry mit exponentiellem Backoff, ConnectionError mit Nachricht, Timeout-Handling
   - Connection-Error-Handling (5xx, DNS, SSL) ✅ DONE: ExchangeAdapterError subclasses (ConnectionError, TimeoutError, RateLimitError, ResponseValidationError), BaseCryptoExchangeAdapter shared retry logic
    - `simulated=False` Tests mit mocked httpx responses ✅ DONE: `TestSimulatedFalseHTTPPath` (12 Tests, httpx.MockTransport — keine echten API-Calls): submit_order-Success für alle 4 Adapter, GET/DELETE-Verb-Routing, signierte Auth auf dem Wire (u. a. HMAC-Recompute Bybit/Binance). Dabei aufgedeckte Live-Mode-Bugfixes: (1) DELETE wurde per POST gesendet — `client.delete`-Branch in Base- und Binance-`_make_signed_request`; (2) Bitget `submit_order` parst `order_id` aus `data[0].orderId` (Live-Shape; simulated-Pfad bleibt flat); (3) `cancel_order`-Success-Detection: Base nutzt top-level `retCode`/`code`, Binance wertet fehlendes Code-Feld als Erfolg
   - ✅ PHASE 9/10/11 COMPLETE: Retry Behavior, HTTP-Error Handling, Schema Validation, Error Class Hierarchy, Dynamic Credential Loading
    - ✅ R5.21–R5.22 COMPLETE: Read/Trade API Separation — `ExecutionConfig` mit `trade_api_key_ref`/`trade_api_secret_ref`/`read_api_key_ref`/`read_api_secret_ref`; `submit_order` prüft TRADE API Keys, `get_order_status` prüft READ API Keys, `cancel_order` prüft TRADE API Keys (write operation); `allowed_exchanges` Whitelist enforced im Pipeline
    - ✅ allowed_exchanges Enforcement: Pipeline step 5b — Order an nicht-listeter Exchange wird REJECTED mit `EXCHANGE_NOT_ALLOWED`
    - ✅ Shadow-Mode-Wiring COMPLETE: `LiveExecutionService` akzeptiert optionalen `ShadowModeLogger`; jede REJECTED/ERROR-Order wird mit vollständigen Request-Parametern (quantity, price, error, run_id) über `ShadowModeLogger.log_rejection()` protokolliert; `ShadowTradeRecord` mit neuem `error`-Feld; Bugfix: `pnl_estimate` liefert 0 für nicht-ausgeführte Orders (war vorher falsches PnL durch fill_price=0); Response-Feld `shadow_mode` kennzeichnet Shadow-Mode-Aktivität
    - ✅ README Section 11 „Systemnutzung“: Arbeitsablauf, API-Endpunkt-Tabelle (Agents/Evolution/Execution/Risk/Research), Shadow Mode, Sicherheitsrichtlinien, Agenten-Lifecycle, Promotion-Kriterien, Regime-Erkennung

3. **Live Execution Safety Gate** — ✅ COMPLETE (R5.3, R5.7, R5.23, R5.24)
    - ✅ Audit-Log (R5.3): optionale `ExecutionLogStore`-Injection in `LiveExecutionService`;
      jeder Trade-Versuch (FILLED/REJECTED/ERROR) wird persistiert (decision_id, timestamp,
      status, order_id, error); Credentials erscheinen nie im Log (R5.20)
    - ✅ Kill-Switch-Monitoring (R5.7): `kill_switch_status()` meldet `enabled`,
      `toggle_count`, `last_toggled_at`; aktiver Kill Switch blockiert Orders sofort
      (Pipeline-Step 1, synchron, < 100 ms); Exchange-Connection-Check folgt mit echter
      Exchange-Integration (im MVP simulated=True)
    - ✅ Maximaler Kapitaleinsatz (R5.23/R5.24): `ExecutionConfig.max_capital`
      (Default `None` → Cap = `min_capital` = 0.01 Einheiten, d. h. standardmäßig nur der
      minimale Test-Betrag); `quantity > max_capital` → REJECTED `MAX_CAPITAL_EXCEEDED`
      (Pipeline-Step 6b)
    - ✅ Safety Gate: `verify_safety_gate()` prüft `kill_switch_present`,
      `min_capital_positive`, `max_capital_valid` und liefert `SafetyGateResult`;
      `activate_live()` ist fail-closed (Live-Aktivierung nur bei bestandenem Gate)
    - 15 neue Tests in `tests/test_execution_safety_gate.py`; Bestands-Tests setzen
      explizit `max_capital=1000.0` (Opt-in für größere Test-Größen); `make check` clean
      (726 Tests, ruff + mypy)

4. **Kill-Switch Auto-Trigger** — ✅ COMPLETE (R5.6)
   - `KillSwitch.record_anomaly(reason)` zählt Exchange-Anomalien der `submit_order`-
     Pipeline (ERROR-Resultat oder Adapter-Exception); bei `auto_trigger_threshold`
     (Default 3) aufeinanderfolgenden Anomalien ohne dazwischenliegende FILLED-Order
     wird der Kill Switch automatisch aktiviert und persistiert
     (`auto_triggered=True`, `trigger_reason`)
   - `KillSwitch.record_success()` (bei FILLED) setzt `anomaly_streak` zurück;
     `deactivate()` (manueller Operator-Neustart) setzt den Streak ebenfalls zurück
   - `KillSwitchConfig` neu: `auto_trigger_enabled` (Default `True`),
     `auto_trigger_threshold` (Default 3), `anomaly_streak`, `auto_triggered`,
     `trigger_reason`; backward-compatible JSON-Load/Save
   - Nur Exception-Typ (nicht `str(e)`) landet in `trigger_reason` —
     Exchange-/System-Details bleiben aus dem persistierten Zustand heraus
   - Response-Feld `kill_switch_auto_triggered` kennzeichnet die auslösende Order
   - 16 neue Tests (`TestKillSwitchAutoTrigger`, `TestLiveExecutionServiceAnomalyAutoTrigger`),
     inkl. Concurrency-Test (10 Threads × 100 Anomalien → genau 1 Trigger); `make check`
     clean (742 Tests, ruff + mypy)
   - Unabhängiges Review (verifiziert: 742 Tests/ruff/mypy reproduziert): **approved**;
     Close-out-Conditions umgesetzt (2 Negative-Tests für Sanitierung + PENDING-Semantik)

5. **Kill-Switch Persistenz-Wiring (bekanntes Safety Gap, vorbestehend)** — ✅ COMPLETE (WI-P5-10)
    - Gap: `routes.py` erzeugte `KillSwitch()` ohne `db_path` → Kill-Switch-Zustand
      (inkl. R5.6 Auto-Trigger UND manueller Aktivierung) ging bei Prozess-Neustart
      verloren (fail-open); nicht-atomarer State-File-Write (`open(..., "w")`) konnte
      bei Crash mid-write die JSON korruptieren → Fallback `enabled=False` (fail-open)
    - Fix: `execution_kill_switch = KillSwitch(db_path=settings.kill_switch_state_path)`
      (neue `Settings`-Option, Default `data/kill_switch.json`, via `.env` überschreibbar);
      `_save_state` schreibt jetzt tmp-Datei + `os.fsync` + atomares `os.replace`
    - Neue `KillSwitch.db_path`-Property (Observability/Wiring-Test); Docstring-Korrektur
      (JSON-, nicht SQLite-Persistenz)
     - 6 neue Tests (manueller + Deaktivierungs-Restart-Roundtrip, Crash-mid-write lässt
       vorherigen Stand intakt, kein tmp-Rest, API-Wiring, API-Neustart-Simulation);
       `make check` clean (750 Tests, ruff + mypy)
     - Unabhängiges Review (verifiziert: 750 Tests/ruff/mypy reproduziert, Invarianten
       I1 Crash-Safety, I2 Restart-Persistenz, I3 keine Sicherheitsgrenzen-Änderung,
       I4 Thread-Safety — alle PASS): **approved**; 6 Findings (3 MINOR, 3 NIT) als
       Folge-Workitems empfohlen: (a) Test-Suite persistiert `enabled: true` ins echte
       `data/kill_switch.json` → nach `make check` startet `make run` mit aktivem Kill
       Switch (fail-closed-Richtung, Test-Isolations-Defekt; Fix: `conftest.py`-Fixture
       mit tmp-Pfad), (b) deterministischer tmp-Dateiname → Multi-Writer-Kollision
       (latent, Single-Writer-Deployments sicher; Fix: `tempfile.mkstemp` +
       `os.unlink` im except), (c) Docker mountet `./data` nicht → Persistenz
       überlebt keine Container-Recreation (Fix: Bind-Mount/Volume + Vermerk)
       — (a) wurde durch WI-P5-11 umgesetzt
     - Gebliebene offene Punkte (bewusst nicht ohne Freigabe geändert):
       `kill_switch_default` (Settings) bleibt unverdrahtet — ein fail-closed First-Start-Default
       wäre eine Verhaltensänderung der Sicherheitsgrenze; `ExecutionLogStore()` in
       `routes.py` hat ebenfalls kein `db_path` (Audit-Log-Persistenz-Wiring, eigenes
       Workitem)

6. **Test-Isolation: API-Tests schreiben nicht in den echten Kill-Switch-State (Review-MINOR-1 von WI-P5-10)** — ✅ COMPLETE (WI-P5-11)
   - Defekt: Der keyless Backward-Compat-Toggle-Test (`test_api_security.py`)
     aktivierte das echte API-Singleton → jeder `make check`-Lauf hinterließ
     `enabled: true` in `data/kill_switch.json` → `make run` danach startete mit
     aktivem Kill Switch (fail-closed-Richtung; Reproduzierbarkeits-Defekt)
   - Fix: neue `tests/conftest.py` mit autouse-Fixture —
     `routes.execution_kill_switch._db_path` wird pro Test auf einen tmp-Pfad
     umgebunden, ein nach dem Test noch aktives Singleton wird deaktiviert;
     Opt-out-Marker `real_kill_switch_state` (in `pyproject.toml` registriert)
     für den Wiring-Test, der den echten konfigurierten Pfad verifiziert
   - 1 neuer Regressionstest (`TestKillSwitchStateIsolation`): ein API-Toggle
     lässt den konfigurierten echten State-Pfad byte-identisch unverändert
     (TDD-Red: vor dem Fix wurde das File bei jedem Lauf umgeschrieben)
    - Empirisch verifiziert: Rest-State-File gelöscht → `make check` (751 Tests)
      erzeugt es NICHT neu — `data/` enthält danach nur `.gitkeep`
    - Unabhängiges Review (verifiziert: 751 Tests/ruff/mypy reproduziert, Isolation
      empirisch in beiden Szenarien — ohne State-File: wird nicht erzeugt, mit
      `enabled:true`-File: byte-identisch (gleiche sha256); Marker-Counterfactual,
      LIFO-Restore und Teardown-Ordnung empirisch belegt; Invarianten I1 keine
      Sicherheitsgrenzen-Änderung, I2 Isolation hält für alle Tests, I3 Wiring-Test
      nicht verwässert, I4 keine Test-Order-Abhängigkeit — alle PASS):
      **approved** (review_id 3); 2 NITs (Marker-Opt-out überspringt den
      Teardown-`deactivate()` — heute unkritisch, einziger Marker-Test rein lesend;
      Assert im `finally` kann Originalexception maskieren — Cleanup ohne harten
      Assert) als Folge-Workitem-Option

7. **Kill-Switch Multi-Writer-Isolation (Review-MINOR-2 von WI-P5-10)** — ✅ COMPLETE (WI-P5-12)
    - Defekt: `_save_state` nutzte einen deterministischen Tmp-Namen
      (`kill_switch.json.tmp`) für alle Writes — zwei `KillSwitch`-Instanzen
      mit geteiltem State-Pfad truncierten/überschrieben sich gegenseitig
      die Tmp-Datei → Lost Updates, gemischte Dokumente, `FileNotFoundError`
      beim `os.replace` (stillschweigend verschluckt) und State-Divergenz
      zwischen In-Memory-Zustand und Datei
    - Fix: `tempfile.mkstemp(dir=<State-Verzeichnis>, prefix=<Name>.", suffix=".tmp")`
      — eindeutige Tmp-Datei pro Writer im selben Dateisystem (`os.replace`
      bleibt atomar); Modus-Erhaltung (`mkstemp` legt 0600 an → `os.chmod`
      auf den State-Datei-Modus bzw. 0644 bei Neuanlage); fehlgeschlagene
      Tmp-Dateien werden best-effort entfernt (kein FD-Leak)
    - 3 neue Tests (`TestKillSwitchMultiWriterIsolation`): deterministischer
      Kollisionstest (erzwungenes Interleaving via `os.fsync`/`os.replace`-
      Blocking; TDD-Red auf altem Code: Datei enthielt B's Dokument, obwohl
      A der zuletzt erfolgreiche Replacer war), Stress (2 Instanzen,
      4 Threads × 25 Toggles: valides JSON, `toggle_count == 50`, keine
      `*.tmp*`-Rückstände, Reload-Konsistenz; 10× stabil), 0644-Modus
      (Neuanlage + Überschreiben)
     - Verifikation (2026-08-20): 754 Tests grün, ruff clean, mypy clean
       (50 source files)
     - Unabhängiges Review (verifiziert: 754 Tests + ruff + mypy exakt reproduziert;
       Counterfactual-Red 5/5 und Green 5/5 via unabhängiges Standalone-Replay der
       erzwungenen Interleaving gegen Pre-Fix `86b8320`; 10× Stabilität 10/10;
       Worktree-Purity + Evidence-Sha256 bestätigt; Invarianten I1 keine
       Sicherheitsgrenzen-/API-/JSON-Format-Änderung, I2 Multi-Writer-Safety
       empirisch bewiesen, I3 Atomicity/Crash-Safety erhalten (FD-Leak-frei,
       Modus-Erhaltung), I4 kein Single-Writer-Verhaltenswechsel (751
       Bestands-Tests grün) — alle PASS): **approved** (review_id 4); 4 NITs
       (nicht blockierend): `path.stat()`-Race → 0644-Fallback akzeptiert
       (korrespondiert mit historischem Umask-Verhalten); Stress-Test
       probabilistisch gegen altes Code (Positionierung in Evidence korrekt);
       kein dir-fsync nach `os.replace` (bestehendes Property, Hardening-
       Kandidat); Red-Run-Exzernt konsistent
 
8. **Docker `./data`-Bind-Mount für Kill-Switch-/Execution-State-Persistenz (Review-MINOR-3 von WI-P5-10)** — ✅ COMPLETE (WI-P5-13)
    - Problem: `docker-compose.yml` mountete nur `./config`, `./prompts`,
      `./schemas` (alle read-only) in den `api`-Container — der in
      `data/kill_switch.json` (`kill_switch_state_path`, relativ zum
      WORKDIR `/app`) persistierende Kill-Switch-State (und ab WI-P5-15
      `data/execution_log.json`) lag damit im Container-Layer und ging bei
      jeder Container-Recreation verloren
    - Fix: genau eine neue Mount-Zeile — `volumes` des `api`-Services erhält
      `- ./data:/app/data` (schreibbar, ohne `:ro`) nach dem `./schemas`-
      Eintrag; bestehende Mounts, Ports, `depends_on`, `env_file` und die
      postgres/redis-Services unverändert; kein Python-Code-/Test-Change
    - README (Abschnitt 5 "Schnellstart"): Vermerk, dass `./data`
      schreibbar nach `/app/data` eingebunden wird, damit Kill-Switch-State
      und Execution-Logs Container-Recreation überleben, und dass der
      Inhalt lokal bleibt (gitignoriert, ausgenommen `.gitkeep`)
    - Verifikation (2026-08-20): `docker compose config --quiet` exit 0
      (ohne Ausgabe); `docker compose build` → `Image smith-api Built`
      (exit 0, Image `sha256:d14826ec009e…`); Smoke-Test: `up -d` →
      `/health` ok (~4 s: `{"status":"ok","live_execution_enabled":false,
      "kill_switch":true}`) → `POST /execution/kill-switch/true` →
      Host-Datei `data/kill_switch.json` zeigt `"enabled": true`
      (`toggle_count: 1`) → `up -d --force-recreate api` → Host-Datei
      byte-identisch (`"enabled": true`, `toggle_count: 1`), frischer
      Container meldet via `/execution/status` `kill_switch: true`
      (persistenter Startzustand) → `POST /execution/kill-switch/false` →
      Host-Datei `"enabled": false` (`toggle_count: 2`) →
      `docker compose down` (ohne `-v`), 0 smith-Container übrig;
      `make check` clean (754 passed, 1 warning in 52.47s; ruff
      `All checks passed!`; mypy `Success: no issues found in 50 source
      files`)
    - Abweichung: Host-Ports 5432/6379 waren durch einen fremden Stack
      (`aurora`) belegt — kein Konflikt, da der smith-Compose-Stack nur
      Port 8080 publiziert (postgres/redis ohne Host-Port-Mapping,
      interne Compose-DNS); Smoke-Test dadurch vollständig durchführbar
    - Prozess-Abweichung: Der Abschlussbericht der ersten Hälfte des
      Implementierungs-Runs enthielt Referenzen auf nicht vorhandene Artefakte
      (behaupteter Commit-SHA, behauptete Review-ID, behauptete
      Evidence-Datei); der Orchestrator verifizierte den Zustand unabhängig
      (damals: kein Commit, keine Evidence, kein Review, Workitem in-progress).
      Build, Smoke-Test, `make check`, Evidence-Erstellung und Commit `8c65a6f`
      wurden in der Fortsetzung desselben Runs tatsächlich ausgeführt — alle
      Verifikationsausgaben in der Evidence stammen aus dieser Fortsetzung
    - Keine Sicherheitsgrenzen-/Verhaltensänderung: Live-Execution bleibt
      deaktiviert, Kill-Switch-Semantik unverändert
    - Review (2026-08-20, `Sisyphus-Junior (independent review)`, Review-ID 5):
      **approved** — I1–I6 alle PASS (reine Config-/Doku-/Evidence-Änderung,
      Live-Execution deaktiviert, Kill-Switch-Semantik unverändert, keine
      Secrets/State-Dateien committet, Stack gestoppt mit erhaltenen Named
      Volumes, Compose-Diff exakt die eine schreibbare Mount-Zeile); NITs
      (nicht blockierend): `kill_switch:true` beim ersten Health-Poll ist die
      korrekte Folge von `kill_switch_default=True` ohne State-Datei
       (fail-safe); bestehende README-Doku-Drift beim Kill-Switch-Endpunkt
       (11.2: `POST /execution/kill-switch` vs. Architektur-Sektion
       `POST /kill-switch`) hier nicht verursacht — Auflösungskandidat für
       ein späteres Doku-/Compose-Workitem

 9. **ExecutionLogStore db_path-Wiring (Audit-Log-Persistenz in `data/execution_log.json`) + conftest-Isolation-Erweiterung** — ✅ COMPLETE (WI-P5-15)
     - Problem: `routes.py` erzeugte `ExecutionLogStore()` ohne `db_path`,
       und beide `LiveExecutionService`-Instanzen (Paper + Crypto) wurden
       ohne `log_store` verdrahtet — Audit-Log-Einträge (R5.3) wurden nie
       in eine JSON-State-Datei geschrieben und gingen bei jedem
       Prozess-Neustart verloren (offener Punkt aus WI-P5-10);
       `_save_state` schrieb nicht-atomar via `open(..., "w")` (Crash
       mid-write konnte die JSON korruptieren); die Test-Isolation
       (WI-P5-11) deckte nur den Kill-Switch-Pfad ab
     - Fix: `execution_log_store = ExecutionLogStore(
       db_path=settings.execution_log_state_path)` (neue `Settings`-Option,
       Default `data/execution_log.json`, via `.env` überschreibbar),
       beide `LiveExecutionService`-Instanzen erhalten
       `log_store=execution_log_store` (Pipeline-Reihenfolge und
       Semantik unverändert, in-memory `get_logs()`-Semantik und die
       `/execution/*`-Endpunkte unverändert); `_save_state` schreibt jetzt
       `tempfile.mkstemp` + `os.fsync` + atomares `os.replace` (spiegelt
       `KillSwitch._save_state`, WI-P5-12: eindeutige Tmp-Datei pro
       Writer, Modus-Erhaltung, best-effort Tmp-Cleanup, kein FD-Leak);
       neue öffentliche `db_path`-Property (spiegelt
       `KillSwitch.db_path`); `_save_state` wird in `add()` unter dem
       Lock aufgerufen (Snapshot-Konsistenz); neue `clear()`-Methode
       (thread-safe, persistiert) für die Test-Isolation; conftest-Autouse-
       Fixture umbindet zusätzlich pro Test
       `routes.execution_log_store._db_path` auf einen tmp-Pfad und ruft
       `clear()` auf (kein In-Memory-Log-State an Folgetests, symmetrisch
       zum Kill-Switch-Teardown); Opt-out-Marker `real_execution_log_state`
       (in `pyproject.toml` registriert)
     - 3 neue Regressionstests (`TestExecutionLogStoreWiring`):
       API-Wiring-Test (`db_path == execution_log_state_path`, mit Marker),
       simulierter Prozess-Neustart (API-Order → neue
       `ExecutionLogStore`-Instanz auf demselben tmp-Pfad lädt den Eintrag
       zurück; TDD-Red: `assert 0 == 1`, die API schrieb ohne `log_store`-
       Wiring nichts in den Store), Test-Isolation (API-Write erzeugt
       kein `data/execution_log.json` im Repo-CWD — Guard-Test, der mit
       dem neuen Write-Pfad relevant wird)
     - Verifikation (2026-08-20): TDD-Red (2 failed:
       `AttributeError: 'ExecutionLogStore' object has no attribute
       'db_path'`; `assert 0 == 1`); danach grün; `make check` clean
       (757 passed, 1 warning in 53.95s; ruff `All checks passed!`; mypy
       `Success: no issues found in 50 source files`); Persistenz-Beweis:
       tmp-Pfad-State geschrieben (exakte JSON in der Evidence-Datei), von
       frischer Instanz geladen (count 1, gleicher Eintrag), keine
       `*.tmp*`-Rückstände; `ls -la data/` vor und nach dem Testlauf:
       keine `execution_log.json`, `kill_switch.json` byte-identisch
       (sha256 `1389df52…`, `enabled: false`, `toggle_count: 2`)
     - Abweichungen: (1) 1 Zeile in `pyproject.toml` (Marker-Registrierung
       `real_execution_log_state`) — liegt außerhalb der Datei-Liste des
       Workitems, ist aber für das symmetrische Pattern nötig (WI-P5-11
       registriert `real_kill_switch_state` dort; ohne Registrierung würde
       pytest `PytestUnknownMarkWarning` melden); (2) README Abschnitt 8:
       Test-Zähler 754 → 757 (Doku-Aktualität desselben Abschnitts)
      - Keine Sicherheitsgrenzen-/Verhaltensänderung: Live-Execution bleibt
        deaktiviert, Kill-Switch-Semantik unverändert, keine
        Risk-Policy-/Whitelist-/Limit-Änderung
       - Review (2026-08-20, `Sisyphus-Junior (independent review)`, Review-ID 6):
         **approved** — I1–I6 alle PASS (Kill-Switch-Code/-Semantik unverändert,
         Live-Execution deaktiviert, Pipeline nur additiv um `log_store=`
         ergänzt, atomares mkstemp+fsync+`os.replace`-Writing mit
         Modus-Erhaltung, keine Secrets/State-Dateien committet,
         Risk-Policy-/Whitelist-/Limit-/Auth unverändert); eigenständiges
         Gate-Replay (757 passed / ruff / mypy clean), `data/kill_switch.json`
         byte-identisch vor/nach dem Lauf; NITs (nicht blockierend,
         Auflösungskandidat für ein späteres Doku-/Hygiene-Workitem): fehlende
         End-zeilenumbrüche in `execution_store.py`/`test_api_execution.py`
         (ruff-konform, kosmetisch), Whitespace-Reflow im Item-8-Text

 10. **NIT-Bundle aus P5-10/P5-11-Reviews: conftest-Teardown bei Marker-Opt-out, finally-Cleanup ohne hartes Assert, Pin-Test korrumpiertes State-File** — ✅ COMPLETE (WI-P5-14)
     - Problem: drei nicht blockierende NITs aus den Reviews zu WI-P5-10
       (Review-ID 2) und WI-P5-11 (Review-ID 3) zur Execution-State-
       Test-Isolation:
       - NIT A (Review-ID 3): "marker-Opt-out überspringt auch den
         Teardown (zukunftssicherungs-relevant)" — der
         `real_execution_log_state`-Marker schaltete die conftest-
         Aufräumung komplett ab; ein Marker-Opt-out-Test direkt hinter
         einem Log-schreibenden Test würde den In-Memory-Log-State des
         Vorgängers erben (heute unkritisch — einziger Marker-Test rein
         lesend; jeder schreibende Marker-Test hätte den Defekt
         ausgelöst)
       - NIT B (Review-ID 2, NIT 4 "Cleanup-Deaktivierung im API-Test
         nicht failure-gesichert (try/finally)" — verbleibender Teil;
         in Review-ID 3 identisch als NIT): das harte
         `assert response.status_code == 200` im `finally`-Cleanup-
         Block von `test_kill_switch_toggle_via_api_leaves_real_state_
         file_untouched` (`test_api_security.py`, einziger
         finally-Assert der Suite — AST-Scan über `tests/`) kann die
         Original-Exception des Tests maskieren
       - NIT C (Review-ID 2, NIT 6): "kein Pin-Test für
         Fail-Open-Fallback bei extern korrumpiertem State-File" —
         `KillSwitch._load_state` fängt `(OSError, json.JSONDecodeError)`
         und fällt auf den Startzustand zurück ("Fallback:
         Startzustand verwenden"); das Verhalten war nicht gepinnt
     - Fix:
       - A: `tests/conftest.py` — Teardown der Autouse-Fixture räumt
         zusätzlich `routes.execution_log_store.clear()` auf, guarded
         durch `request.node.get_closest_marker(
         "real_execution_log_state") is None` (Guard zwingend: nur
         ohne Marker ist der Store auf den tmp-Pfad gebunden; ein
         ungeguardetes Clear würde in die echte State-Datei
         `data/execution_log.json` persistieren und die Isolation
         brechen; wirkt vor der Monkeypatch-Rücksetzung, also auf dem
         tmp-Pfad); Setup-Clear und Kill-Switch-Teardown unverändert
       - B: `tests/test_api_security.py` — hartes Assert aus dem
         `finally`-Block entfernt (Cleanup-POST
         `client.post("/execution/kill-switch/False")` bleibt);
         best-effort-Cleanup ohne hartes Assert, damit ein
         Cleanup-Fehler die Original-Exception nicht maskiert
       - C: `tests/test_kill_switch.py` — neue Klasse
         `TestCorruptedStateFileFallback` (1 Test, 3 nummerierte
         Assertion-Blöcke): korrumpiertes State-File
         (`{not valid json!!`) → Konstruktor ohne Exception,
         Fail-Open-Fallback auf Startzustand in beiden Richtungen
         (`enabled=False` → inaktiv, `enabled=True` → aktiv), Folge-
         `deactivate()` repariert das File zu gültigem JSON mit
         `enabled == False`; dazu neue Klasse
         `TestIsolationFixtureTeardown` (2 Tests, bewusst
         Datei-Reihenfolge-abhängig): Regressionstest für NIT A —
         Test 1 (ohne Marker) schreibt einen Log-Eintrag in den
         tmp-gebundenen Store, Test 2 (Marker-Opt-out) direkt danach
         sieht einen leeren Store (TDD-Red ohne Fix: `assert 1 == 0`)
     - Verifikation (2026-08-20): TDD-Red vor dem conftest-Fix
       (`test_opt_out_test_sees_clean_log_state` rot mit
       `assert 1 == 0` — In-Memory-Rest von Test 1; Pin-Test C
       bereits grün, da er bestehendes Verhalten pinnt und auch vor
       jeder Quelländerung bestehen muss); danach 3/3 grün;
       `make check` clean (760 passed, 1 warning in 56.31s; ruff
       `All checks passed!`; mypy `Success: no issues found in 50
       source files`, exit 0); finally-Assert-AST-Scan über den
       ganzen `tests/`-Baum: pre-fix genau 1 Treffer
       (`tests/test_api_security.py:354`), post-fix 0; `data/` vor
       und nach dem Gate byte-identisch (`.gitkeep` +
       `kill_switch.json`, 208 Bytes, sha256
       `1389df52f9a2e125a05a2ee96b13263870a234236506d2487c12cbc06d2383a9`,
       `enabled: false`, `toggle_count: 2`), keine
       `data/execution_log.json` (vollständige verbatim-Ausgaben in
       `evidence/wi-p5-14-nit-bundle-test-evidence.md`)
     - Abweichungen (im Workitem-Scope deklariert, Details in der
       Evidence-Datei): (1) `tests/test_api_security.py` liegt
       außerhalb der Workitem-Dateiliste, trägt aber den NIT B
       (einziger finally-Assert der Suite); (2) `README.md`
       Test-Zähler 757 → 760 (gleiche Deklarationsabweichung wie
       WI-P5-15)
     - Keine Sicherheitsgrenzen-/Verhaltensänderung: der
       Fail-Open-Fallback wird GEpinnt, nicht geändert (eine
       Fail-Closed-Semantik wäre eine Sicherheitsgrenzen-Änderung
       und erfordert explizite Freigabe); Live-Execution bleibt
        deaktiviert; Kill-Switch-Semantik unverändert
      - Review (2026-08-20, `Sisyphus-Junior (independent review)`, Review-ID 7):
        **approved** — I1–I6 alle PASS (keine src/-Änderungen, Fail-Open-Fallback
        gepinnt und nicht geändert, Live-Execution deaktiviert, keine
        Secrets/State-Dateien committet, Conftest-Teardown-Guard mit korrekter
        Richtung/Platzierung, keine Risk-Policy-/Whitelist-/Limit-/Auth-/
        Pipeline-Änderung); eigenständiges Gate-Replay (760 passed / ruff /
        mypy clean), `data/kill_switch.json` byte-identisch vor/nach dem Lauf,
        0 finally-Asserts im tests/-Baum; NIT (nicht blockierend, in der
        Evidence dokumentiert): fehlender End-zeilenumbruch in
        `tests/test_kill_switch.py` (kosmetisch, ruff-konform)

   See `docs/spec-phase5-live-execution.md` und `docs/phase5-epic.md` für den definierten Umfang.

Live-Execution bleibt standardmäßig deaktiviert — keine Änderungen an Sicherheitsgrenzen ohne explizite Freigabe.

- **Outcome Generator persistence migration** — moved `OutcomeGenerator` from `tests/_test_utils.py` to `src/trading_harness/services/outcome_generator.py`; `OutcomeGenerator` accepts `OutcomeStore` protocol (default: `InMemoryOutcomeStore`); `PersistedOutcomeStore` in `outcome_store.py` already implements the protocol; tests: 18 tests in `test_outcome_generator.py`
- **Structured agent output queries via API routes** — 4 new endpoints: `GET /agent/analyses`, `GET /agent/analyses/run/{run_id}`, `GET /agent/analyses/agent/{agent_id}`, `GET /agent/analyses/snapshot/{snapshot_id}` — wired to `PersistedAgentAnalysisStore`
- **Bugfix: ParameterMutation** — fixed random attribute selection (was ~1/6 chance of correct attr), dead key (`weighting_strategies` vs `weighting_strategy`), and conditional generation increment — all `MutationType` → `ParameterMutation` mappings now pass explicit `attr` parameter

Phase 2 — Evaluation: ✅ EvaluationService results persistence wired — `PersistedEvaluationResultStore` added, injected into `EvaluationService` via `routes.py`, `add()` called after every `evaluate_agent()`, test suite with 8 tests, `make check` clean.

Phase 3 — Evolution: ✅ COMPLETE. 8 core services, 6 test files, 24 API endpoints wired.

Services:
- `AgentGenomeStore` — in-memory genome persistence (CRUD, filtering by category/status/generation)
- `PersistedAgentGenomeStore` — PostgreSQL-backed variant with schema migration
- `AgentFactory` — mutation strategies (INDICATOR_ADD, INDICATOR_REMOVE, TIMEFRAME_ADD,
  TIMEFRAME_REMOVE, PARAMETER_MUTATION, TEMPERATURE_MUTATION, RECOMBINATION, SPECIALIZATION,
  SIMPLIFICATION, DIVERSITY_INJECTION)
- `ChallengerPool` — champion/challenger pairing, promotion evaluation, promotion execution,
  demotion, category stats
- `HallOfFame` — top-performer tracking with max-entry limit, category filtering, best lookup
- `Graveyard` — retired/rejected agent tracking with category filtering
- `EvolutionService` — orchestrator tying factory + pool + hall-of-fame + graveyard + rollback
- `PromotionPolicy` — minimum_observations >= 10, relative_improvement_min >= 0.03,
  out-of-sample, walk-forward, shadow mode, ensemble contribution, security gates

Models added to `models.py`:
- `MutationType` enum (10 mutation types)
- `GenomeMutation` — mutation record with type, changes, and hypothesis
- `ChampionChallenger` — pair record for evaluation
- `EvolutionRun` — promotion/retirement run record
- `PromotionDecision` — decision with approved/rejected and reasons
- `RollbackEntry` — status transition audit record
- `HallOfFameRecord` / `GraveyardRecord` — lifecycle archives

Test suites (1193 lines total):
- `test_agent_genome_store.py` — 15 tests (CRUD, filtering, hash)
- `test_agent_factory.py` — 10+ tests (all mutation strategies, recombination)
- `test_challenger_pool.py` — 10+ tests (pairing, promotion eval, demotion, stats)
- `test_hall_of_fame.py` — 9 tests (add, sorting, limits, lookups)
- `test_graveyard.py` — 6 tests (add, retrieve, category filter)
- `test_evolution_service.py` — 18+ tests (mutant generation, recombination, challenger mgmt,
  promotion decisions, retire/reject, probation, rollback)

API endpoints added to `routes.py` (24 new endpoints):
- `POST /evolution/mutate` — generate mutant from parent
- `POST /evolution/recombine` — recombine two parents
- `POST /evolution/challengers/{agent_id}/add` — add challenger to pool
- `GET /evolution/challengers/pairs/{category}` — list champion/challenger pairs
- `POST /evolution/challengers/evaluate` — evaluate promotion criteria
- `POST /evolution/challengers/promote` — execute promotion
- `POST /evolution/challengers/demote` — demote to probation
- `POST /evolution/hall-of-fame` — add to hall of fame
- `GET /evolution/hall-of-fame` — list all hall of fame records
- `GET /evolution/hall-of-fame/{category}` — list by category
- `GET /evolution/hall-of-fame/top/{category}` — get best in category
- `POST /evolution/graveyard` — add to graveyard
- `GET /evolution/graveyard` — list all graveyard records
- `GET /evolution/graveyard/{category}` — list by category
- `GET /evolution/promotion-history/{category}` — list promotion runs
- `GET /evolution/rollbacks` — list all rollbacks
- `GET /evolution/rollbacks/{agent_id}` — list rollbacks for agent
- `POST /evolution/rollback` — rollback agent status
- `GET /evolution/population-stats/{category}` — category population stats
- `GET /evolution/population-stats` — all category stats

Address the security and correctness gaps documented in the README development sequence before
adding paper- or live-trading capabilities. Live execution remains out of scope.

## Resume

```bash
git status -sb
./scripts/bootstrap.sh
make check
opencode
```

Inside OpenCode, run `/resume` to reconstruct context from Git and continue the next safe task.

## Last verification

- `make check` (2026-08-20, WI-P4-4-Fix `f89a742`): 784 Tests grün, ruff clean, mypy clean (51 source files); `data/kill_switch.json` byte-identisch (sha256 `1389df52f9a2e125a05a2ee96b13263870a234236506d2487c12cbc06d2383a9`)
- `make check` (2026-08-20, Review-9/13-Fixes `e109400`/`4e4cd38`): 772 Tests grün, ruff clean, mypy clean (50 source files); `data/kill_switch.json` byte-identisch (sha256 `1389df52f9a2e125a05a2ee96b13263870a234236506d2487c12cbc06d2383a9`)
- `make check` (2026-08-20, WI-P5-12 Multi-Writer-Isolation): 754 Tests grün, ruff clean, mypy clean (50 source files)
- `make check` (2026-08-20, WI-P5-11 Test-Isolation): 751 Tests grün, ruff clean, mypy clean (50 source files); `data/kill_switch.json` wird von der Suite nicht mehr erzeugt
- `make check` (2026-08-19, WI-P5-10 Kill-Switch Persistenz-Wiring): 750 Tests grün, ruff clean, mypy clean (50 source files)
- `make check` (2026-08-19, R5.6 Kill-Switch Auto-Trigger): 742 Tests grün, ruff clean, mypy clean (50 source files)
- `make check` (2026-08-19, Safety Gate): 726 Tests grün, ruff clean, mypy clean (50 source files)
- `./scripts/bootstrap.sh --check`
- `docker compose config --quiet`
- locked Docker image build and `/health` smoke test