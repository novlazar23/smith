# Security-Review-Checklist — Phase 5 (Live Execution)

**Zweck:** Verbindliche Security-Review-Checklist für alle Phase-5-Sicherheitskontrollen.
Dieses Dokument ist das im Epic benannte Deliverable von **WI-P5-8**
(`docs/phase5-epic.md:161` — „`docs/security-review-phase5.md` — new (Security Review
Checklist)“) und schließt die in Review #18 (Sisyphus-Junior-R3, 2026-08-20,
`changes-requested`) dokumentierte Blocking-Gap.

**Verifikationsstand:** Alle `file:line`-Referenzen wurden am **HEAD `09f8f21`**
(2026-08-20) durch direktes Lesen der Quelldateien verifiziert. Testnamen und
Testanzahlen wurden per `pytest --collect-only` (2026-08-20) bestätigt.

**Legende:**

- **PASS** = Kontrolle in Code verifiziert, Default-Wert bestätigt, Testabdeckung vorhanden.
- **PASS (dokumentierte Einschränkung)** = Kontrolle funktioniert wie dokumentiert; eine
  bekannte, ausdrücklich dokumentierte Einschränkung ist unter „Befunde“ aufgeführt.

---

## 1. Kill Switch (R5.5–R5.7)

| # | Kontrolle | Implementierung (verifiziert) | Default / Config-Key | Test-Evidenz | Status |
|---|---|---|---|---|---|
| 1.1 | Thread-sicherer Zustand | `src/trading_harness/services/kill_switch.py:38` (`threading.Lock`), `is_active()` :163–166, `activate()` :136–146, `deactivate()` :148–161 | — | `tests/test_kill_switch.py` (32 Tests): `test_concurrent_activate_deactivate`, `test_concurrent_writers_stress`, `test_is_active_performance` | PASS |
| 1.2 | Atomare JSON-Persistenz | `kill_switch.py:70–134` (`_save_state`): `tempfile.mkstemp` :91–93, `os.chmod` :98, `fsync` :115, `os.replace` :116 | State-Pfad: `data/kill_switch.json` (Config-Key `kill_switch_state_path`, `src/trading_harness/config.py:21`) | `test_crash_before_replace_keeps_previous_state`, `test_multi_writer_collision_no_lost_update`, `test_no_tmp_remains_after_successful_save`, `test_state_file_mode_0644_new_and_overwrite` | PASS |
| 1.3 | Zustand überlebt Prozess-Neustart | Laden: `kill_switch.py:43–68` (`_load_state`, korrumpiertes File → Init-State :66–68); API-Wiring: `src/trading_harness/api/routes.py:605` (`KillSwitch(db_path=settings.kill_switch_state_path)`) | — | `test_persist_and_load`, `test_manual_activation_survives_restart`, `test_deactivation_survives_restart`, `test_corrupted_state_file_falls_back_to_init_state`; `tests/test_api_execution.py`: `kill_switch_state_survives_process_restart` | PASS |
| 1.4 | R5.6 Auto-Trigger: 3 aufeinanderfolgende Exchange-Fehler (ohne FILLED) aktivieren den Kill Switch | Trigger-Logik: `kill_switch.py:173–198` (`record_anomaly`, Schwellenwert-Prüfung :186); Streak-Reset bei Erfolg: `kill_switch.py:200–205`; Pipeline-Anbindung: `src/trading_harness/services/live_execution_service.py:347–354` (FILLED → `record_success` :350, ERROR → `record_anomaly` :352); Exception-Pfad: :370–375 | `auto_trigger_enabled: bool = True` (`kill_switch.py:22`), `auto_trigger_threshold: int = 3` (`kill_switch.py:23`) | `test_trigger_at_threshold`, `test_no_trigger_below_threshold`, `test_success_resets_streak`, `test_auto_trigger_disabled`, `test_persistence_of_auto_trigger`, `test_concurrent_anomalies_trigger_exactly_once`; `tests/test_live_execution_service.py`: `three_consecutive_errors_auto_trigger`, `two_errors_do_not_trigger`, `filled_resets_error_streak`, `exception_path_counts_as_anomaly` | PASS |
| 1.5 | API-Toggle hinter Trade-API-Key | `routes.py:673–683` (`POST /execution/kill-switch/{enabled}` mit `dependencies=[Depends(require_trade_key)]`, `activate()`/`deactivate()` :679–682) | — | `tests/test_api_execution.py`: `toggle_kill_switch`; `tests/test_api_security.py`: `test_kill_switch_toggle_wrong_key`, `test_kill_switch_toggle_no_key`, `test_kill_switch_toggle_via_api_leaves_real_state_file_untouched` | PASS |
| 1.6 | Konstruktor-Default bei fehlender State-Datei | `kill_switch.py:36` (`enabled: bool = False`), `_load_state` kehrt ohne File zum Init-State zurück (`kill_switch.py:45–46`) | `enabled = false` (beobachteter Zustand in `data/kill_switch.json` am 2026-08-20: `enabled=false`, `auto_trigger_threshold=3`, `auto_triggered=false`) | `test_initial_state_disabled`; dokumentiert in Review #15 (F3) und Review #18 (Invariant a) | PASS |
| 1.7 | Kill Switch blockiert die Pipeline (Schritt 2) | `live_execution_service.py:162–173` (Rejection `KILL_SWITCH_ACTIVE`, vor Rate-Limit/Dedup/Exchange) | — | `test_live_execution_service.py`: `kill_switch_blocks_order`, `paper_adapter_rejected_with_kill_switch`; `tests/test_api_execution.py`: `submit_order_kill_switch`; `tests/test_execution_safety_gate.py`: `test_active_kill_switch_still_blocks_orders` | PASS |
| 1.8 | Monitoring-Zustand (R5.7) | `live_execution_service.py:442–444` (`kill_switch_status()`), Konfig-Snapshot `kill_switch.py:207–220` | — | `tests/test_execution_safety_gate.py`: `test_kill_switch_status_reports_initial_state`, `test_kill_switch_status_reflects_toggle` | PASS |

## 2. Rate Limiting (R5.10)

| # | Kontrolle | Implementierung (verifiziert) | Default / Config-Key | Test-Evidenz | Status |
|---|---|---|---|---|---|
| 2.1 | Token Bucket in N/min-Semantik: Burst = Limit, Refill = Limit/60 Tokens/s | `src/trading_harness/services/rate_limiter.py:20` (Konstruktor), :23 (Start-Tokens = Limit → Burst), `_refill` :28–48 (globale Refill-Rate `limit/60` :39, pro Symbol :43), Kapazitäts-Deckel :40–42, :45–47 | `global_rate_limit: int = 10`, `symbol_rate_limit: int = 2` (`live_execution_service.py:44–45`) | `tests/test_rate_limiter.py` (14 Tests): `test_refill_scales_with_limit`, `test_symbol_refill_scales_with_limit`, `test_refill_does_not_exceed_capacity`, `test_sustained_throughput_per_minute`, `test_symbol_sustained_rate_per_minute` | PASS |
| 2.2 | Globales + pro-Symbol-Bucket | `rate_limiter.py:50–78` (`allow(symbol)`: Global-Check :64, pro-Symbol-Check :68–73, Token-Verbrauch :76–77), `RLock` :25 | 10/min global, 2/min pro Symbol (Defaults) | `test_allow_when_under_limit`, `test_refuses_when_global_exceeded`, `test_refuses_when_symbol_exceeded`, `test_concurrent_requests`, `test_symbol_limit_concurrent`, `test_global_limit`, `test_symbol_limit`, `test_reset_all`, `test_reset_symbol` | PASS |
| 2.3 | Limiter wird aus Config gebaut (wirkt im Standard-Pfad) | `live_execution_service.py:119–122` (Default-Limiter aus `ExecutionConfig`) | — | `tests/test_live_execution_service.py`: `default_rate_limiter_uses_execution_config`, `default_config_rate_limit_enforced` | PASS |
| 2.4 | Enforcement in der Pipeline (Schritt 3) | `live_execution_service.py:175–186` (Rejection `RATE_LIMIT_EXCEEDED`) | — | `rate_limit_blocks_order`, `paper_adapter_rejected_with_rate_limit`; `tests/test_api_execution.py`: `submit_order_rate_limit` | PASS |

## 3. Order-Deduplizierung

| # | Kontrolle | Implementierung (verifiziert) | Default / Config-Key | Test-Evidenz | Status |
|---|---|---|---|---|---|
| 3.1 | Duplikat-Erkennung über `decision_id` + Symbol + Side | `src/trading_harness/services/order_deduplicator.py:20–22` (Key `decision_id:symbol:side`), :24–45 (`is_duplicate`, Check :37) | — | `tests/test_order_deduplicator.py` (11 Tests): `test_first_order_not_duplicate`, `test_same_order_is_duplicate`, `test_different_decision_not_duplicate`, `test_different_symbol_not_duplicate`, `test_different_side_not_duplicate` | PASS |
| 3.2 | Memory-bounded mit periodischem Trim | `order_deduplicator.py:15` (`max_entries: int = 10000`), :17 (`deque(maxlen=…)`), Trim-Auslösung :41–44, `_trim` :47–51, `clear` :53–72 | `max_entries = 10000` | `test_concurrent_duplicate_detection`, `test_concurrent_different_orders`, `test_max_entries_limit`, `test_clear_all`, `test_clear_decision`, `test_seen_count_increments` | PASS |
| 3.3 | Enforcement in der Pipeline (Schritt 4) | `live_execution_service.py:188–199` (Rejection `DUPLICATE_DECISION_ID`) | — | `duplicate_rejected`, `paper_adapter_duplicate_decision_rejected`; `tests/test_api_execution.py`: `submit_order_dedup` | PASS |

## 4. Symbol- & Exchange-Whitelist (R5.6, R5.21)

| # | Kontrolle | Implementierung (verifiziert) | Default / Config-Key | Test-Evidenz | Status |
|---|---|---|---|---|---|
| 4.1 | Symbol-Whitelist; **leer = alle erlaubt** (dokumentiertes Default, README §11.4) | `live_execution_service.py:201–212` (Rejection `SYMBOL_NOT_WHITELISTED` nur, wenn Liste nicht leer) | `symbol_whitelist: list[str] = []` (`live_execution_service.py:51`); API-Wiring setzt das Feld nicht (`routes.py:606–609`) → Default allow-all | `tests/test_live_execution_service.py`: `whitelisted_symbol_passes`, `non_whitelisted_symbol_rejected`, `empty_whitelist_allows_all` | PASS |
| 4.2 | Exchange-Whitelist; leer = alle erlaubt | `live_execution_service.py:214–225` (Rejection `EXCHANGE_NOT_ALLOWED`) | `allowed_exchanges: list[str] = []` (`live_execution_service.py:50`) | `allowed_exchange_passes`, `empty_allowed_exchanges_allows_all`, `allowed_exchanges_blocks_unlisted_exchange`, `allowed_exchanges_allows_listed_exchange`, `allowed_exchanges_empty_allows_any`; `tests/test_api_execution.py`: `crypto_submit_with_exchange_name` | PASS |
| 4.3 | Zweites Symbol-Gate über Risk Policy (scharfer als die Whitelist) | `config/risk-policy.yaml:5–7` (`allowed_symbols: BTCUSDT, ETHUSDT`), Enforce `src/trading_harness/services/risk_engine.py:14–15` (`SYMBOL_NOT_ALLOWED`); wird in der Pipeline-Re-Check-Schleife aktiv (`live_execution_service.py:261–292`) | `allowed_symbols` aus `config/risk-policy.yaml` | `risk_engine_rejects_unknown_symbol` (`tests/test_live_execution_service.py::TestRiskEngineIntegration`) | PASS |

## 5. Kapital-Limits (R5.23/R5.24)

| # | Kontrolle | Implementierung (verifiziert) | Default / Config-Key | Test-Evidenz | Status |
|---|---|---|---|---|---|
| 5.1 | Minimum-Capital pro Order | `live_execution_service.py:227–238` (Rejection `MIN_CAPITAL_NOT_MET`) | `min_capital: float = 0.01` (`live_execution_service.py:46`) | `min_capital_enforced`, `min_capital_passes`, `paper_adapter_min_capital_enforced`; `tests/test_api_execution.py`: `submit_order_min_capital`, `crypto_submit_min_capital_rejected` | PASS |
| 5.2 | Maximum-Capital pro Order, Default = min_capital | `live_execution_service.py:240–256` (Rejection `MAX_CAPITAL_EXCEEDED`), effektives Limit :241–245 (`None` → `min_capital`) | `max_capital: float \| None = None` → effektiv `0.01` (`live_execution_service.py:47–48`) | `tests/test_execution_safety_gate.py`: `test_quantity_above_max_capital_rejected`, `test_quantity_within_max_capital_passes`, `test_default_cap_equals_min_capital`, `test_minimal_test_amount_allowed_by_default` | PASS |

## 6. Hebel- & Tagesverlust-Limits (R5.1–R5.4)

| # | Kontrolle | Implementierung (verifiziert) | Default / Config-Key | Test-Evidenz | Status |
|---|---|---|---|---|---|
| 6.1 | Maximaler Tagesverlust | `risk_engine.py:23–24` (Rejection `MAX_DAILY_LOSS_REACHED`) | `max_daily_loss: 0.02` (= 2 %, `config/risk-policy.yaml:10`) | `tests/test_risk_engine.py` (4 Tests): `test_valid_trade_is_approved`, `test_kill_switch_rejects`, `test_excess_leverage_rejects`, `test_low_rr_rejects`; Policy-Laden `routes.py:610` | PASS |
| 6.2 | Maximaler Hebel | `risk_engine.py:17–18` (Rejection `MAX_LEVERAGE_EXCEEDED`) | `max_leverage: 2.0` (`config/risk-policy.yaml:12`) — siehe Befund **F3** zur README-Angabe „1.0x“; die Pipeline-Re-Check-Instanz fordert fix Hebel 1.0 (`live_execution_service.py:272`; `TradeProposal`-Default `models.py:102`) | `test_excess_leverage_rejects` | PASS |
| 6.3 | Max. Risiko pro Trade → Positionslimit | `risk_engine.py:46–58` (`max_risk_per_trade` → `risk_fraction` → `max_position_size = risk_amount / stop_distance` :57–58); Cap in Pipeline: `live_execution_service.py:294–296` | `max_risk_per_trade: 0.005` (`config/risk-policy.yaml:9`) | `test_valid_trade_is_approved`; `quantity_capped_to_max_position_size` (`tests/test_live_execution_service.py::TestMaxPositionSizeEnforcement`) | PASS |
| 6.4 | Weitere deterministische Limits: Positionsanzahl, Portfoliorisiko, Slippage, R:R | `risk_engine.py:20–21` (`MAX_POSITIONS_REACHED`), :26–27 (`MAX_PORTFOLIO_RISK_REACHED`), :29–30 (`MAX_SLIPPAGE_EXCEEDED`), :38–44 (`MINIMUM_RISK_REWARD_NOT_MET`), :35–36 (`INVALID_STOP_DISTANCE`) | `max_portfolio_risk: 0.04`, `max_positions: 5`, `minimum_risk_reward: 1.8`, `max_slippage_bps: 20` (`config/risk-policy.yaml:11,13,14,15`) | `test_low_rr_rejects` | PASS |

## 7. Deterministische Risk Engine — keine LLM-Überschreibung

| # | Kontrolle | Implementierung (verifiziert) | Default / Config-Key | Test-Evidenz | Status |
|---|---|---|---|---|---|
| 7.1 | LLM-Ausgaben dürfen deterministische Limits nicht überschreiben | README.md §2 („Nicht verhandelbare Architekturregeln → Deterministische Sicherheitsgrenzen: … LLMs dürfen diese Werte lesen und berücksichtigen, aber nicht überschreiben“) und §11.4 („Keine dieser Grenzen darf von LLM-Ausgaben überschrieben werden. Deterministische Policy hat Vorrang.“); Code-Pfad: keine LLM-Eingabe in den `/execution/*`-Routen (einfache dict-Payloads, `routes.py:657–670`); deterministische Re-Check-Schleife `live_execution_service.py:258–292`; Legacy `/risk/evaluate` ebenfalls nur über die deterministische Risk Engine (`routes.py:117–119`) | — | Code-Lektüre (keine LLM-Kopplung an Execution-Routen); Review #15 (Invariant c) und Review #18 (Invariant c) bestätigen | PASS |
| 7.2 | Kill-Switch-Check innerhalb der Risk Engine | `risk_engine.py:11–12` (Rejection `KILL_SWITCH_ACTIVE`) | — | `test_kill_switch_rejects` | PASS |

## 8. Network Policy (R5.15–R5.17)

| # | Kontrolle | Implementierung (verifiziert) | Default / Config-Key | Test-Evidenz | Status |
|---|---|---|---|---|---|
| 8.1 | Endpoint-Whitelist; leer = alle erlaubt (dokumentiert) | `src/trading_harness/services/network_policy.py:38–43` (Konstruktor, Patterns), `is_allowed` :45–64 (leere Whitelist → `True` :47–48), API-Wiring `routes.py:613` | `network_allowed_patterns: list[str] = []` (`config.py:31`) | `tests/test_network_policy.py` (12 Tests): `test_empty_policy_allows_all`, `test_allows_whitelisted_url`, `test_blocks_non_whitelisted_url`, `test_blocks_unrelated_exchange`, `test_allows_wildcard_pattern` | PASS |
| 8.2 | VerletZungen werden protokolliert (Audit) | `network_policy.py:52–58` (Violation-Record + Warning-Log), `violation_count` :66–69, `get_violations` :71–73 | — | `test_violation_logged`, `test_multiple_violations`, `test_violation_count_property`, `test_get_violations_with_limit` | PASS |
| 8.3 | Enforcement in der Pipeline (Schritt 9) | `live_execution_service.py:298–311` (Rejection `NETWORK_POLICY_VIOLATION`), Exchange-URL-Mapping `:388–402` | — | `network_policy_allows_matching_url`, `network_policy_blocks_violation` (`tests/test_live_execution_service.py::TestNetworkPolicyIntegration`) | PASS |

## 9. Credential-Check & Read/Trade-API-Separation (R5.18–R5.22)

| # | Kontrolle | Implementierung (verifiziert) | Default / Config-Key | Test-Evidenz | Status |
|---|---|---|---|---|---|
| 9.1 | Credential-Verwaltung ohne Raw-Value-Logging | `src/trading_harness/services/credential_manager.py:48–64` (`get`: Secret Store → env, Cache), `summary()` :81–89 (nur Referenzen, keine Werte) | `credential_source: str = "env"` (`config.py:34`) | `tests/test_credential_manager.py` (11 Tests): `test_get_returns_env_value`, `test_get_returns_none_when_missing`, `test_is_configured_returns_true/false`, `test_summary_no_raw_values`, `test_clear_cache`, `test_get_caches_value` | PASS |
| 9.2 | Trade-Credential-Check (Submit, Schritt 10) | `live_execution_service.py:313–327` (Rejection `TRADE_CREDENTIALS_NOT_CONFIGURED`) | Key-Referenzen `TRADE_API_KEY`/`TRADE_API_SECRET` (`live_execution_service.py:53–54`) | `credentials_configured_passes`, `credentials_missing_rejected`, `missing_api_key_rejected` (`tests/test_live_execution_service.py::TestCredentialManagerIntegration`) | PASS |
| 9.3 | Read/Trade-Separation: Read-Operationen erfordern Read-Credentials, Writes Trade-Credentials | `get_order_status`: `live_execution_service.py:564–569` (`READ_CREDENTIALS_NOT_CONFIGURED`); `cancel_order` (Write): :589–594 (`TRADE_CREDENTIALS_NOT_CONFIGURED`) | `READ_API_KEY`/`READ_API_SECRET`-Referenzen (`live_execution_service.py:55–56`) | `tests/test_live_execution_service.py::TestReadTradeApiSeparation` (7 Tests): `submit_requires_trade_credentials`, `read_order_status_requires_read_credentials`, `cancel_order_requires_trade_credentials`, `…_succeeds`-Varianten, `config_has_custom_key_refs` | PASS |
| 9.4 | API-Auth: fehlender Key → 401, falscher Key → 403; Header oder Query-Param | `src/trading_harness/api/security.py:18–32` (`_verify_key`: 401 :23–27, 403 :28–32), `require_read_key` :35–44 (Header `X-Read-API-Key` / Query `read_api_key`), `require_trade_key` :47–56 (Header `X-Trade-API-Key` / Query `trade_api_key`) | `read_api_key: str = ""`, `trade_api_key: str = ""` (`config.py:27–28`) | `tests/test_api_security.py` (17 Tests): `test_key_from_header_valid`, `test_key_from_header_invalid`, `test_key_missing`, `test_order_wrong_key_rejected`, `test_order_missing_key_rejected`, `test_order_correct_key_accepted`, `test_status_with_read_key`, `test_logs_with_read_key` | PASS |
| 9.5 | Bekannte Einschränkung: **fail-open, wenn kein Key konfiguriert ist** | `security.py:41–42` (read) und :53–54 (trade): leerer Config-Key → Zugriff ohne Auth. Als backward-kompatible Entscheidung dokumentiert: Docstrings in `security.py:38` bzw. :50, `docs/handoff.md:136` („Backward-compatible: Wenn Key nicht konfiguriert, werden Endpoints durchgelassen“); Review #15 (F1): „Not a blocking gap for the epic criteria, but should be tightened before any live-enablement“ | Default `""` (nicht konfiguriert) | `test_no_key_configured_allows_access`, `test_order_without_key_config`, `test_status_without_key_config`, `test_kill_switch_toggle_without_key_config` | PASS (dokumentierte Einschränkung, s. Befund F1) |

## 10. Live Execution standardmäßig deaktiviert + Safety Gate

| # | Kontrolle | Implementierung (verifiziert) | Default / Config-Key | Test-Evidenz | Status |
|---|---|---|---|---|---|
| 10.1 | Live Execution per Default aus | `config.py:19` (`live_execution_enabled: bool = False`), `.env.example:14` (`LIVE_EXECUTION_ENABLED=false`), `ExecutionConfig` :43 (Default `False`), Service-State :128 | `LIVE_EXECUTION_ENABLED=false` | `submit_order_disabled`, `crypto_submit_live_disabled` (`tests/test_api_execution.py`), `default_disabled`, `submit_when_disabled` (`tests/test_live_execution_service.py::TestLiveExecutionServiceBasic`) | PASS |
| 10.2 | Pipeline-Schritt 1 blockiert bei deaktivierter Live Execution | `live_execution_service.py:149–160` (Rejection `LIVE_EXECUTION_DISABLED`) | — | `submit_order_disabled` (oben); `paper_adapter_disabled_rejected` | PASS |
| 10.3 | Fail-closed `activate_live()`: nur bei bestandenem Safety Gate | `live_execution_service.py:446–456` (`activate_live` setzt `_enabled=True` nur, wenn `verify_safety_gate().ready`), `deactivate_live` :458–461 | — | `tests/test_execution_safety_gate.py::TestSafetyGate`: `test_activate_live_is_fail_closed_when_gate_fails`, `test_activate_live_enables_when_gate_passes`, `test_deactivate_live_still_works` | PASS |
| 10.4 | Safety-Gate-Checks | `live_execution_service.py:404–440`: `kill_switch_present` :408–416, `min_capital_positive` :418–425, `max_capital_valid` :427–436 | — | `test_verify_safety_gate_passes_with_safe_defaults`, `test_verify_safety_gate_fails_on_nonpositive_min_capital`, `test_verify_safety_gate_fails_on_nonpositive_max_capital` | PASS |
| 10.5 | Crypto-Adapter standardmäßig simuliert; ohne Credentials immer simuliert | `src/trading_harness/services/crypto_exchange_adapter.py:221` (`simulated: bool = True`), :242 (`self._simulated = simulated or not (api_key and api_secret)`); Router meldet `LIVE` nur bei Key **und** Secret: `routes.py:829–834` | `simulated=True` | `crypto_status_shows_credentials` (`tests/test_api_execution.py::TestCryptoExecutionEndpoints`); README §8 („simulated=True standardmäßig bei allen Crypto-Adaptern“) | PASS |
| 10.6 | Legacy `ExecutionGateway` fail-closed | `src/trading_harness/services/execution_gateway.py:10–16` (Default `live_enabled=False`, Rejection `LIVE_EXECUTION_DISABLED`), Instanz mit `settings.live_execution_enabled` (`routes.py:46`) | `False` | `tests/test_execution_gateway.py` (Legacy-Suite, Teil der 772 Tests) | PASS |

## 11. ExecutionLogStore (R5.3)

| # | Kontrolle | Implementierung (verifiziert) | Default / Config-Key | Test-Evidenz | Status |
|---|---|---|---|---|---|
| 11.1 | JSON-Persistenz mit atomarem Write | `src/trading_harness/services/execution_store.py:84–139` (`_save_state`: `mkstemp` :105–107, `fsync` :122, `os.replace` :123), Laden :64–82 | State-Pfad: `data/execution_log.json` (Config-Key `execution_log_state_path`, `config.py:22`) | `tests/test_execution_store.py` (11 Tests): `test_persist_and_load`, `test_atomic_write_crash_integrity`, `test_corrupted_state_file_fallback` | PASS |
| 11.2 | In-Memory-Fallback ohne Persistenz-Pfad | `execution_store.py:53–57` (`db_path=None` → kein Datei-I/O, `_save_state` :97–98) | `db_path=None` → in-memory | Basis-Tests `test_add_log`, `test_get_all`, `test_get_by_decision_id`, `test_get_by_run`, `test_count` (ohne State-File) | PASS |
| 11.3 | Keine Credentials in Logs/Persistenz | `execution_store.py:115–119` (Felder mit „key“ im Namen werden beim Serialize entfernt), Klassen-Docstring :50 | — | `test_no_credentials_in_logs`; `tests/test_execution_safety_gate.py`: `test_credentials_never_persisted_to_store` | PASS |
| 11.4 | ID-Erzeugung vollständig im Lock (Review-13, N2) | `execution_store.py:157–159` (ID `exec-{ms}-{len}` innerhalb `with self._lock`) | — | `test_concurrent_adds` | PASS |
| 11.5 | Jeder Trade-Versuch wird geloggt (R5.3), API-Instanz persistiert | Pipeline-Hook: `live_execution_service.py:515–525` (`log_store.add`); API-Wiring: `routes.py:603` (`ExecutionLogStore(db_path=settings.execution_log_state_path)`) | — | `tests/test_execution_safety_gate.py::TestAuditLogPersistence`: `test_filled_order_persisted_to_store`, `test_rejected_order_persisted_to_store`, `test_error_result_persisted_to_store`; `tests/test_api_execution.py::TestExecutionLogStoreWiring`: `execution_log_store_wired_with_state_path`, `execution_log_state_survives_process_restart`, `api_writes_do_not_touch_real_execution_log_path` | PASS |

## 12. Shadow Mode

| # | Kontrolle | Implementierung (verifiziert) | Default / Config-Key | Test-Evidenz | Status |
|---|---|---|---|---|---|
| 12.1 | Entscheidungen werden geloggt, nicht ausgeführt | `src/trading_harness/services/shadow_mode_logger.py:100–141` (`log_order` mit simuliertem Fill), `log_rejection` :143–174 (kein Fill, Status `REJECTED` :164); PnL = 0 für Nicht-Fills :43–52 | Slippage `0.0005` (0,05 %), Commission `0.001` (0,1 %) (`shadow_mode_logger.py:64–65`) | `tests/test_shadow_mode_logger.py` (17 Tests): `test_slippage_calculation`, `test_commission_calculation`, `test_pnl_estimate`, `test_log_rejection_creates_rejected_record`, `test_rejected_record_has_zero_pnl`, `test_summary_counts_rejections`, `test_get_records_filters` | PASS |
| 12.2 | `ShadowModeAdapter` (ExchangeAdapter-Konform, Fallback bei Fehler) | `shadow_mode_logger.py:229–287` (Delegation :258–262, `ExchangeAdapterError` → Shadow-Logger :263–267) | — | `test_shadow_submit_order`, `test_shadow_get_balance`, `test_shadow_get_ticker`, `test_shadow_order_in_logger` | PASS |
| 12.3 | Pipeline loggt REJECTED/ERROR im Shadow Mode | `live_execution_service.py:527–537` (`log_rejection` bei Status `REJECTED`/`ERROR`), Response-Flag `shadow_mode` :549 | — | `tests/test_live_execution_service.py::TestLiveExecutionServiceShadowMode`: `rejected_order_logged_to_shadow_mode`, `kill_switch_rejection_shadow_logged`, `successful_order_not_shadow_logged`, `no_shadow_logger_flag_false` | PASS |
| 12.4 | Shadow-API-Endpunkte (submit hinter Trade-Key, Lese-Endpunkte hinter Read-Key) | `routes.py:707–729` (`POST /execution/shadow/submit`, Trade-Key), :732–738 (`GET /execution/shadow/summary`, Read-Key), :741–756 (`GET /execution/shadow/records`, Read-Key) | — | Endpoint-Wiring per Code-Lektüre verifiziert (HEAD 09f8f21); Logger-Verhalten vollständig in `tests/test_shadow_mode_logger.py` (17 Tests) abgedeckt; Auth-Verhalten generisch in `tests/test_api_security.py` (17 Tests) | PASS |

## 13. Secrets-Handling

| # | Kontrolle | Implementierung (verifiziert) | Default / Config-Key | Test-Evidenz | Status |
|---|---|---|---|---|---|
| 13.1 | `.env` wird nie committet | `.gitignore:1` (`.env`) | — | `git check-ignore .env` → ignored (verifiziert 2026-08-20) | PASS |
| 13.2 | `data/*` gitignored, ausgenommen `.gitkeep` | `.gitignore:15–16` (`data/*`, `!data/.gitkeep`) | — | `git check-ignore data/kill_switch.json data/execution_log.json` → ignored; `git ls-files data/` → nur `.gitkeep` (verifiziert 2026-08-20) | PASS |
| 13.3 | `.env.example` ist eine Non-Secret-Vorlage | `.env.example:1–17` (nur Platzhalter wie `LLM_API_KEY=change-me`; keine Exchange-Keys/Secrets) | `LIVE_EXECUTION_ENABLED=false` (:14), `KILL_SWITCH_DEFAULT=true` (:15) | Dateilektüre (HEAD 09f8f21) | PASS |
| 13.4 | Credentials tauchen nie in Logs/Audit/Anomalie-Reasons auf | `credential_manager.py:81–89` (Summary ohne Rohwerte), `execution_store.py:115–119` (Key-Felder entfernt), `live_execution_service.py:371–374` (Anomalie-Reason nur Exception-Typ, keine Message) | — | `test_summary_no_raw_values`, `test_no_credentials_in_logs`, `test_credentials_never_persisted_to_store`, `exception_reason_sanitized` | PASS |

## 14. Docker-Data-Volume

| # | Kontrolle | Implementierung (verifiziert) | Default / Config-Key | Test-Evidenz | Status |
|---|---|---|---|---|---|
| 14.1 | `./data` wird schreibbar nach `/app/data` eingebunden; Config/Prompts/Schemas read-only | `docker-compose.yml:14–18` (`./config:/app/config:ro` :15, `./prompts:/app/prompts:ro` :16, `./schemas:/app/schemas:ro` :17, `./data:/app/data` :18 — ohne `:ro`) | — | `docker compose config --quiet` → clean (verifiziert 2026-08-20); Review #5 (WI-P5-13, Invarianten I1–I6 PASS) | PASS |
| 14.2 | Kill-Switch- und Execution-Log-State überleben Container-Recreation | State-Dateien liegen unter `/app/data` via `kill_switch_state_path`/`execution_log_state_path` (`config.py:21–22`), verdrahtet in `routes.py:603,605` | `data/kill_switch.json`, `data/execution_log.json` | Dateiebene: `kill_switch_state_survives_process_restart`, `execution_log_state_survives_process_restart` (`tests/test_api_execution.py`); Container-Recreation empirisch in Review #5 (I1–I6) verifiziert | PASS |

---

## Befunde (nicht blockierend, dokumentiert)

| ID | Befund | Referenz |
|---|---|---|
| F1 | **API-Auth fail-open bei nicht konfiguriertem Key**: Sind `trade_api_key`/`read_api_key` leer (Default `""`), sind die Execution-Endpunkte unauthentifiziert. Als backward-kompatible Entscheidung dokumentiert (`security.py:38,50,41–42,53–54`; `docs/handoff.md:136`). Vor einer Live-Aktivierung ist die Auth zu verschärfen. | Review #15 (F1) |
| F2 | **Legacy unauthentifizierte Routen** (vor Phase 5, seit Initial-Commit, außerhalb des Phase-5-Scopes): `POST /kill-switch/{enabled}` (`routes.py:132–136` — flippt nur das Legacy-Modul-Flag für `/health` und `/risk/evaluate`; „true“ ist die sichere Richtung; berührt den persistenten Phase-5-KillSwitch nicht) und `POST /execution/orders/{decision_id}` (`routes.py:127–129` — Legacy `ExecutionGateway`, fail-closed). Empfehlung: Follow-up-WI zum Entfernen oder Auth-Guard. | Review #15 (F2) |
| F3 | **README/Risk-Policy-Diskrepanz beim Hebel**: README §11.4 nennt „Max Hebel 1.0x“, die Policy legt `max_leverage: 2.0` fest (`config/risk-policy.yaml:12`). Die Pipeline-Re-Check-Instanz fordert fix Hebel 1.0 (`live_execution_service.py:272`). Die README-Angabe ist unpräzise; die wirksame Policy-Grenze ist 2.0. | Code vs. README §11.4 |
| F4 | **„kill_switch default true“** (README §11.4) bezeichnet das Legacy-Config `kill_switch_default` (`config.py:20`, genutzt in `routes.py:47` und `/risk/evaluate` `routes.py:119`). Der persistente Phase-5-KillSwitch startet ohne `data/kill_switch.json` mit `enabled=false` (`kill_switch.py:36`). Primäres Gate bleibt `live_execution_enabled=false` — keine Order kann ohne Live-Aktivierung ausgeführt werden. | Review #15 (F3), Review #18 (a) |
| F5 | **Zweites Symbol-Gate**: Die Risk-Policy `allowed_symbols` (`config/risk-policy.yaml:5–7`: BTCUSDT, ETHUSDT) wird in der Pipeline-Re-Check-Schleife erzwungen (`risk_engine.py:14–15`), zusätzlich zur (leeren) Symbol-Whitelist. Für Live-Betrieb auf weitere Symbole ist die Policy zu erweitern, nicht die Engine. | Code-Lektüre (HEAD 09f8f21) |

---

## Review-Nachweis

Unabhängige Closeout-Reviews (Harness-Review-IDs 1–24, abgelesen aus der Harness-DB
`/root/.harness/projects/smith/harness.db`, read-only, 2026-08-20):

| Review-ID | Arbeitspaket | Reviewer | Verdict | Datum (UTC) |
|---|---|---|---|---|
| 1 | WI-P5-9 | sisyphus-junior | approved | 2026-08-19 |
| 2 | WI-P5-10 | Sisyphus-Junior | approved | 2026-08-19 |
| 3 | WI-P5-11 | sisyphus-junior | approved | 2026-08-20 |
| 4 | WI-P5-12 | Sisyphus-Junior | approved | 2026-08-20 |
| 5 | WI-P5-13 | Sisyphus-Junior | approved | 2026-08-20 |
| 6 | WI-P5-15 | Sisyphus-Junior | approved | 2026-08-20 |
| 7 | WI-P5-14 | Sisyphus-Junior | approved | 2026-08-20 |
| 8 | WI-P5-1 | Sisyphus-Junior | approved | 2026-08-20 |
| 9 | WI-P5-2 | Sisyphus-Junior | changes-requested | 2026-08-20 |
| 10 | WI-P5-3 | Sisyphus-Junior | approved | 2026-08-20 |
| 11 | WI-P5-4 | Sisyphus-Junior | approved | 2026-08-20 |
| 12 | WI-P5-5 | Sisyphus-Junior | approved | 2026-08-20 |
| 13 | WI-P5-6 | Sisyphus-Junior | changes-requested | 2026-08-20 |
| 14 | WI-P5-2 (Re-Review, Fix `e109400`) | Sisyphus-Junior-R6 | approved | 2026-08-20 |
| 15 | WI-P5-7 | Sisyphus-Junior-R3 | approved | 2026-08-20 |
| 16 | WI-P5-6 (Re-Review, Fix `4e4cd38`) | Sisyphus-Junior-R6 | approved | 2026-08-20 |
| 17 | WI-P4-2 | Sisyphus-Junior-R4 | approved | 2026-08-20 |
| 18 | WI-P5-8 | Sisyphus-Junior-R3 | changes-requested | 2026-08-20 |
| 19 | WI-P4-3 | Sisyphus-Junior-R4 | approved | 2026-08-20 |
| 20 | WI-P4-5 | Sisyphus-Junior-R5 | approved | 2026-08-20 |
| 21 | WI-P4-4 | Sisyphus-Junior-R4 | changes-requested | 2026-08-20 |
| 22 | WI-P4-6 | Sisyphus-Junior-R5 | approved | 2026-08-20 |
| 23 | WI-P4-7 | Sisyphus-Junior-R5 | approved | 2026-08-20 |
| 24 | WI-P4-1 | Sisyphus-Junior-R3 | approved | 2026-08-20 |

**Zu Review #18 (WI-P5-8, `changes-requested`):** Alle übrigen Akzeptanzkriterien von
WI-P5-8 waren PASS; die einzige Blocking-Gap war das fehlende Deliverable
`docs/security-review-phase5.md` (Akzeptanzkriterium 2: „Security Review Checklist
erstellt“, `docs/phase5-epic.md:161,165`). **Dieses Dokument schließt diese Gap.**
Die Findings F1–F4 entsprechen den in Review #15 (F1–F3) bzw. Review #18 (a)
beschriebenen, als nicht blockierend eingestuften Punkten; F5 ist eine neue,
klarstellende Beobachtung dieser Checklist.

**Verifikations-Baseline:**

- `make check` am **HEAD `09f8f21`** (2026-08-20): **772 Tests passed, 0 failures**
  (1 bestehende `StarletteDeprecationWarning`), `ruff check` clean,
  `mypy` clean (50 source files).
- `pytest --collect-only` (2026-08-20): 217 Tests in den 11 Phase-5-Security-Testdateien
  (`test_kill_switch.py` 32, `test_live_execution_service.py` 61, `test_api_execution.py`
  27, `test_shadow_mode_logger.py` 17, `test_execution_safety_gate.py` 17,
  `test_rate_limiter.py` 14, `test_network_policy.py` 12, `test_order_deduplicator.py` 11,
  `test_execution_store.py` 11, `test_credential_manager.py` 11, `test_risk_engine.py` 4).
- `docker compose config --quiet`: clean (2026-08-20).
- `data/kill_switch.json` und `data/execution_log.json` sind gitignored
  (`.gitignore:15–16`); `git ls-files data/` zeigt nur `data/.gitkeep`.
