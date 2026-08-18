# Development Handoff

## Current state

No implementation task is currently in progress. The repository provides a reproducible Python 3.12
environment, a frozen dependency lock, matching local/CI checks, and versioned OpenCode rules and
commands. The complete German OpenCode usage and cross-device workflow is documented in
`docs/opencode-nutzung.md`.

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

Phase 5 — Live Execution: ✅ CORE SERVICES + Read/Trade API Auth + Crypto Adapters + Shadow Mode + Network Isolation + Credential Management + Minimal Capital COMPLETE.
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
   - `simulated=False` Tests mit mocked httpx responses 🚧 TODO: No real API calls (simulated=True only for safety)
   - ✅ PHASE 9/10/11 COMPLETE: Retry Behavior, HTTP-Error Handling, Schema Validation, Error Class Hierarchy, Dynamic Credential Loading
   - ✅ R5.21–R5.22 COMPLETE: Read/Trade API Separation — `ExecutionConfig` mit `trade_api_key_ref`/`trade_api_secret_ref`/`read_api_key_ref`/`read_api_secret_ref`; `submit_order` prüft TRADE API Keys, `get_order_status` prüft READ API Keys, `cancel_order` prüft TRADE API Keys (write operation); `allowed_exchanges` Whitelist enforced im Pipeline
   - ✅ allowed_exchanges Enforcement: Pipeline step 5b — Order an nicht-listeter Exchange wird REJECTED mit `EXCHANGE_NOT_ALLOWED`

3. **Live Execution Safety Gate** — explizite Freigabe erforderlich
   - Audit-Log aller Live-Transaktionen
   - Kill Switch Monitoring (wirkliche Exchange-Verbindung)
   - Maximaler Kapitaleinsatz auf MinCapital beschränkt (ExecutionConfig.min_capital)

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

- `./scripts/bootstrap.sh --check`
- `docker compose config --quiet`
- locked Docker image build and `/health` smoke test