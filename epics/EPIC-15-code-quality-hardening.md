# EPIC-15: Code Quality Hardening — Fix all pyright errors

## Problem
Nach EPIC-01 bis EPIC-14 (inkl. Multi-Exchange, Backtesting) existieren 62 pyright-Fehler im Codebase. Alle sind pre-existing — keine kommen von EPIC-13 oder EPIC-14. Diese Fehler blockieren die Definition of Done der technischen Spezifikation (Abschnitt 33), die einen unabhängigen Review ohne kritische Mängel erfordert.

**Gesamtzahl: 62 pyright errors, 0 warnings, 0 informations.**

## Ziel
Alle 62 pyright-Typfehler beheben und durch Tests verifizieren, dass der gesamte `packages/`-Directory pyright-clean ist.

## Abhängigkeiten
- EPIC-01 bis EPIC-14 (alle vorangehenden Epics)
- Keine Abhängigkeiten untereinander außer WP06/WP07 nach den Fixes

## Definition of Done
1. `pyright packages/` meldet 0 errors
2. Alle `AgentReport`-Konstruktionen enthalten `expected_return` und `calibrated_confidence`
3. Alle `max(dict.get(...))`-Anti-Patterns durch korrekte `max(d, key=d.get)` ersetzt
4. Backtesting-Typen (datetime/NaT, Series, possibly unbound) korrigiert
5. Governance-State-Machine None-sicher
6. Observability-Imports auflösbar oder gekapselt
7. Keine ndarray→float Type-Mismatches

## Arbeitspakete

### WP01: Agent Report Schema Fixes (17 errors)
**Dateien:** anomaly_agent.py, chart_agent.py, contrainer/agent.py, cross_market_agent.py, elliott_agent.py, fibonacci_agent.py, historical_analogy_agent.py, indicator_agent.py, multi_timeframe/agent.py, news_agent.py, orderflow_agent.py, regime_agent.py

**Fehlerkategorie:** `reportCallIssue` — `AgentReport`-Instanzen fehlen die Parameter `expected_return` und `calibrated_confidence`. Das Schema (packages/schemas/agent_report.py) definiert diese als optionale Felder, aber pyright prüft die Positional-Parameter und verlangt sie.

**Fix-Strategie:**
- Alle `AgentReport(...)` Calls ergänzen um `expected_return=None` (oder berechneten Wert) und `calibrated_confidence=None` (oder `raw_confidence`)
- `contrainer/agent.py:189` — `_contrarian_opposite` map: Key ist `VoteDirection` Enum, aber der Lookup-Parameter `direction` ist `str`. Fix: direction als str behandeln oder Mapping auf StrEnum umstellen.
- `multi_timeframe/agent.py:100` — `Counter[str] += float` (weight ist float, Counter erwartet int). Fix: Counter durch `defaultdict(float)` ersetzen oder auf float zählen.
- `multi_timeframe/agent.py:107` — `max(direction_votes, key=direction_votes.get)` → Key-Function muss callable sein, Counter.get gibt 0 zurück wenn key fehlt, funktioniert aber mit Float-Werten.

### WP02: Dict max() Pattern Fixes (6 errors)
**Dateien:** historical_analogy_agent.py, multi_timeframe/agent.py, regime/hmm.py, regime/rules.py

**Fehlerkategorie:** `reportCallIssue` — `max(dict.get())` aufrufen funktioniert nicht, weil `max()` einen iterable oder `key=`-Parameter erwartet, aber `dict.get()` einen einzelnen Wert zurückgibt.

**Fix-Strategie:**
- `historical_analogy_agent.py:448,540` — `max(probability, key=probability.get)` → korrekte Syntax für max mit key-Funktion
- `multi_timeframe/agent.py:107` — `max(direction_votes, key=direction_votes.get)` → Key-Function muss callable sein
- `regime/hmm.py:137` — `max(scores, key=scores.get)` → korrekte Syntax
- `regime/rules.py:113` — `max(scores, key=scores.get)` → korrekte Syntax

### WP03: Backtesting Package Fixes (11 errors)
**Dateien:** core.py, datafeed.py, engine.py, strategies.py, validation.py

**Fehlerkategorie:** `reportAttributeAccessIssue`, `reportIncompatibleMethodOverride`, `reportArgumentType`, `reportPossiblyUnboundVariable`

**Fix-Strategie:**
- `core.py:116` — `self._drawdown` nicht definiert. Fix: `@property` mit `getattr(self, '_drawdown', 0.0)` oder `_drawdown` als dataclass field hinzufügen
- `core.py:189` — `model_validator` überschreibt `validate` von BaseModel. Fix: Validator-Name ändern oder `@model_validator` korrekt verwenden
- `datafeed.py:110,208,220` — `pd.Timestamp(...).to_pydatetime()` kann `NaTType` zurückgeben. Fix: auf `NaT` prüfen. Series-Typisierung bei `float(row["..."])` durch `float(row["..."] or 0)` ergänzen.
- `engine.py:127` — `candle` möglicherweise ungebunden. Fix: `candle` als `Candle | None` typisieren und prüfen.
- `strategies.py:114,190` — `on_bar` Rückgabetyp `StrategySignal | None`, aber BaseStrategy verlangt `StrategySignal`. Fix: Rückgabetyp erweitern oder immer Signal zurückgeben.
- `validation.py:275,276` — `datetime.fromisoformat(candles[idx].timestamp)` — `candles[idx].timestamp` ist bereits `datetime`, nicht `str`. Fix: `.isoformat()` aufrufen oder direkt verwenden.

### WP04: Governance/State Machine Fixes (3 errors)
**Dateien:** state_machine.py

**Fehlerkategorie:** `reportArgumentType`, `reportOperatorIssue`

**Fix-Strategie:**
- `state_machine.py:113` — `previous_state=None` bei `log_state_transition` für initial registration. Fix: `previous_state` Parameter als `str | None` typisieren.
- `state_machine.py:138,182` — `to_state not in allowed` bzw. `to_state in allowed` wo `allowed` vom Typ `frozenset[AgentState] | None` sein kann. Fix: `if allowed is None or to_state not in allowed:` pattern.

### WP05: Observability/Misc Fixes (17 errors)
**Dateien:** config.py, mlflow_client.py, tracing.py, pipeline.py, second_round.py, clickhouse/engine.py, sqlalchemy/repository.py, streaming/base.py, uncertainty/confidence.py, domain/news/clustering.py

**Fehlerkategorie:** `reportIncompatibleMethodOverride`, `reportPrivateImportUsage`, `reportMissingImports`, `reportCallIssue`, `reportArgumentType`, `reportReturnType`

**Fix-Strategie:**
- `config.py:25` — `json: bool` Field überschreibt `BaseModel.json()`. Fix: Field in `json_format` umbenennen.
- `mlflow_client.py:129` — `mlflow.sklearn.log_model` verwendet nicht-exportiertes `sklearn`. Fix: `import sklearn` Guards.
- `mlflow_client.py:157,159` — `mlflow.search_runs` gibt `list[Run] | DataFrame` zurück, aber Funktion gibt `list[Any]` an. Fix: Union-Typ erweitern oder cast.
- `tracing.py:92,98` — Imports von `opentelemetry.trace.sampling` und `opentelemetry.exporter.otlp.proto.grpc.trace_exporter` können nicht auflösen. Fix: `try/except ImportError` Guards.
- `tracing.py:168` — `force_flush(timeout_millis=5000)` — Parameter heißt `timeout` oder `force_flush` API. Fix: korrekten Parameter-Namen verwenden.
- `pipeline.py:162,192` — `seal_records` Typisierung `list[dict]` vs `list[SealRecord]`. Fix: Union-Typ oder Casting.
- `second_round.py:285` — `vote_distribution` mit String-Keys ("long", "short") aber Typ ist `dict[VoteDirection, float]`. Fix: VoteDirection-Enum-Werte verwenden.
- `clickhouse/engine.py:214` — `httpx.post(url, data=query)` — `data` erwartet `RequestData | None` (dict), nicht `str`. Fix: `content=query` verwenden.
- `sqlalchemy/repository.py:63` — `entity.updated_at` unbekanntes Attribut auf `Base*`. Fix: `hasattr`-Check oder TypeVar erweitern.
- `streaming/base.py:141` — `async def list_dead_events` muss `list[dict[str, Any]]` zurückgeben, aber kein `return`. Fix: `return []` am Ende.
- `uncertainty/confidence.py:65-67` — `max(lower, 0.0)` wo `lower` `ndarray | float` sein kann. Fix: `float(max(lower, 0.0))` oder `float(np.maximum(lower, 0.0))`.
- `domain/news/clustering.py:44,51` — `NewsCluster` in `list[NewsEvent]` appended, Rückgabetyp `list[NewsCluster]` aber Liste ist `list[list[NewsEvent]]`. Fix: Cluster als `NewsCluster`-Instanzen bauen.
- `historical_analogy_agent.py:224,226` — `ndarray | None` in return type `dict[str, NDArray[float64]]`. Fix: None-Werte als leere Arrays oder Float-Arrays casten.

### WP06: Tests — Verification of Zero Pyright Errors (0 errors, verification)
**Dateien:** tests/unit/test_agent_report_fixes.py, tests/unit/test_backtesting_fixes.py, tests/unit/test_governance_fixes.py, tests/unit/test_observability_fixes.py, tests/unit/test_misc_fixes.py, tests/pyright_verification.py

**Aufgaben:**
- Tests für alle Agent Report Schema Fixes schreiben
- Backtesting-Fixes mit Tests abdecken
- Governance state machine None-Safety testen
- Observability import guards testen
- Verifizierung: `pyright packages/` meldet 0 errors

### WP07: Documentation — Evidence File (0 errors)
**Dateien:** .harness/projects/smith/evidence/EPIC-15-code-quality-evidence.md

**Aufgaben:**
- Pyright-Fehlerkategorien dokumentieren
- Before/After pyright output speichern
- Fix-Strategie dokumentieren
- Test-Ergebnisse verknüpfen

## Risiken
1. **Regression bei AgentReport-Änderungen** — expected_return/calibrated_confidence ändern Semantik, muss mit Konsens-Pipeline kompatibel sein
2. **MLflow-Import-Guards** — wenn MLflow nicht installiert, muss graceful degradation funktionieren
3. **Strategies.on_bar return type** — Ändern von `StrategySignal` auf `StrategySignal | None` in BaseStrategy ist breaking change

## Rollback
- Jeder Fix ist isoliert (einzelne Dateien)
- Git revert pro WP möglich
- Pyright-Check als Regressionstest vor Commit
