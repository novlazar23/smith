# Phase 5 — Live Execution

## 1. Problem

Das System hat Research (Phase 1), Evaluation (Phase 2), Evolution (Phase 3) und Paper
Trading (Phase 4) implementiert. Trading-Entscheidungen werden jedoch nie gegen eine
tatsächliche Exchange durchgereicht.

Es fehlt ein **technisch isolierter, standardmäßig deaktivierter Execution Service**, der
Trading-Entscheidungen deterministisch gegen eine Exchange-Adapter-Schicht durchreicht —
unter strikter Einhaltung aller Sicherheitsgrenzen.

## 2. Anforderungen

### 2.1 Execution Service

**R5.1** Muss eine `LiveExecutionService`-Klasse bereitstellen, die `TradeDecision`-Objekte
empfängt und gegen die Risk Policy validiert, bevor sie an den Exchange Adapter weitergeleitet
werden.

**R5.2** Muss standardmäßig deaktiviert sein (`live_execution_enabled: false` in Config).
Explizite Aktivierung nur über Config-Flag.

**R5.3** Muss jeden Trade-Versuch (erfolgreich, abgelehnt, fehlgeschlagen) im Audit Log
protokollieren mit: `decision_id`, `timestamp`, `action`, `result`, `error` (falls).

**R5.4** Muss eine `ExchangeAdapter`-Abstraktion definieren (polymorphes Interface), die
verschiedene Exchange-Protokolle unterstützen kann (z.B. CCXT-basiert), aber nicht CCXT selbst
ist. Kein Exchange wird im MVP fest integriert.

### 2.2 Kill Switch

**R5.5** Muss einen globalen `KillSwitch` bereitstellen, der sofort alle laufenden und
ausstehenden Trades stoppt.

**R5.6** Muss sowohl manuell aktivierbar (API-Call) als auch automatisch auslösbar sein
(beim ersten Anomalie-Ereignis).

**R5.7** Muss innerhalb von 100ms nach Aktivierung alle Execution-Vorgänge blockieren.

**R5.8** Muss den Zustand (`enabled`/`disabled`) persistieren und bei Prozess-Neustart
wiederherstellen.

### 2.3 Rate Limits

**R5.9** Muss pro-Symbol und globale Order-Rate-Limits enforced haben.

**R5.10** Muss konfigurierbare Limits unterstützen (Standard: 10 Orders/Minute global,
2 Orders/Minute pro Symbol).

**R5.11** Muss bei Überschreitung die Order ablehnen mit Reason `RATE_LIMIT_EXCEEDED`.

### 2.4 Order Deduplication

**R5.12** Muss Order-Deduplizierung basierend auf `decision_id + symbol + side` durchführen.

**R5.13** Muss bei Duplikaten die Order ablehnen mit Reason `DUPLICATE_DECISION_ID`.

**R5.14** Muss Duplikat-Erkennung thread-sicher implementieren (auch bei parallelen Calls).

### 2.5 Network Isolation

**R5.15** Muss eine Endpoint-Whitelist unterstützen, die definiert, welche Exchange-Endpoints
angefragt werden dürfen.

**R5.16** Muss alle anderen Netzwerk-Ausgänge blockieren (konfigurierbare Policy).

**R5.17** Muss Network-Policy-Verletzungen im Audit Log protokollieren.

### 2.6 Credential Management

**R5.18** Muss eigene Credential-Verwaltung haben (getrennt von LLM- und Risk Engine Credentials).

**R5.19** Muss Credentials nur aus env vars oder einem externen Secret Store lesen.

**R5.20** Muss Credentials **niemals** in Logs, Audit oder Debug-Output schreiben.

### 2.7 Read/Trade API Separation

**R5.21** Muss separate Auth für Trade-Aktionen haben vs. Lese-Zugriffe.

**R5.22** Muss jede Trade-Aktion mit einem separaten API-Key authentifizieren.

### 2.8 Minimal Capital

**R5.23** Muss standardmäßig nur minimale Test-Beträge erlauben (configurable, Standard: 0.01
Einheiten).

**R5.24** Muss Position Sizing auf das minimale Kapital begrenzen.

## 3. Architektur

```
┌────────────────────────────────────────────────────────────┐
│                  Live Execution Flow                        │
│                                                             │
│  TradingRun (DECISION)                                      │
│       │                                                     │
│       ▼                                                     │
│  ExecutionService                                           │
│       │                                                     │
│       ├──► KillSwitch (check, <100ms)                       │
│       │                                                     │
│       ├──► Rate Limiter (global + per-symbol)               │
│       │                                                     │
│       ├──► Order Deduplicator (decision_id + symbol + side) │
│       │                                                     │
│       ├──► Risk Policy Re-check (symbol whitelist, etc.)    │
│       │                                                     │
│       └──► ExchangeAdapter (polymorph interface)            │
│               │                                             │
│               └──► Network Policy (endpoint whitelist)      │
│                                                             │
├────────────────────────────────────────────────────────────┤
│              Data Layer                                     │
│                                                             │
│  PostgreSQL:                                                │
│  ┌──────────────────────────────────────────────┐          │
│  │ execution_log (id, decision_id, result, ...) │          │
│  │ execution_config (kill_switch, rate_limits)  │          │
│  └──────────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────┘
```

## 4. Datenmodelle

### 4.1 ExecutionLog

```python
class ExecutionStatus(StrEnum):
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"

class ExecutionLog(BaseModel):
    id: str = Field(default_factory=lambda: f"exec-{uuid4()}")
    decision_id: str
    run_id: str
    symbol: str
    side: str
    status: ExecutionStatus
    order_id: str | None = None
    error: str | None = None
    timestamp: datetime = Field(default_factory=utcnow)
```

### 4.2 ExecutionConfig

```python
class ExecutionConfig(BaseModel):
    live_execution_enabled: bool = False
    kill_switch_enabled: bool = False
    global_rate_limit: int = 10  # orders per minute
    symbol_rate_limit: int = 2   # orders per minute per symbol
    allowed_endpoints: list[str] = Field(default_factory=list)
    min_capital: float = 0.01
    credentials_config: dict = Field(default_factory=dict)  # ref only, never values
```

## 5. Algorithmen

### 5.1 Kill Switch

```python
class KillSwitch:
    def __init__(self, enabled: bool = False):
        self._enabled = enabled
        self._lock = threading.Lock()

    def activate(self) -> None:
        with self._lock:
            self._enabled = True

    def deactivate(self) -> None:
        with self._lock:
            self._enabled = False

    def is_active(self) -> bool:
        with self._lock:
            return self._enabled
```

### 5.2 Rate Limiter (Token Bucket)

```python
class RateLimiter:
    def __init__(self, global_limit: int, symbol_limit: int):
        self._global_tokens = asyncio.Semaphore(global_limit)
        self._symbol_tokens: dict[str, asyncio.Semaphore] = {}
        self._symbol_limit = symbol_limit

    def allow(self, symbol: str) -> bool:
        # Try global
        if not self._global_tokens.acquire(blocking=False):
            return False
        # Try per-symbol
        sem = self._get_semaphore(symbol)
        if not sem.acquire(blocking=False):
            self._global_tokens.release()
            return False
        return True
```

### 5.3 Order Deduplication

```python
class OrderDeduplicator:
    def __init__(self):
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def is_duplicate(self, decision_id: str, symbol: str, side: str) -> bool:
        key = f"{decision_id}:{symbol}:{side}"
        with self._lock:
            if key in self._seen:
                return True
            self._seen.add(key)
            return False
```

## 6. Fehlerbehandlung

**F5.1** Kill Switch aktiv → Alle Orders sofort REJECTED.

**F5.2** Rate Limit überschritten → Order REJECTED mit `RATE_LIMIT_EXCEEDED`.

**F5.3** Duplikat erkannt → Order REJECTED mit `DUPLICATE_DECISION_ID`.

**F5.4** Exchange Adapter Error → Order REJECTED mit Fehler-Details, Audit-Log-Eintrag.

**F5.5** Credential nicht gesetzt → Execution Service startet, aber keine Orders möglich.

**F5.6** Network Policy Violation → Order REJECTED, Audit-Log-Eintrag, optional Alert.

## 7. Security

**S5.1** Live Execution standardmäßig deaktiviert (`live_execution_enabled: false`).

**S5.2** Keine Credentials in Logs/Audit.

**S5.3** Kill Switch muss zuverlässig sein (thread-safe, schnell).

**S5.4** Order Deduplication muss thread-safe sein.

**S5.5** Read/Trade API Separation durch separate Auth.

**S5.6** Alle Sicherheitsgrenzen der Risk Engine werden vor Execution再次 geprüft.

## 8. Teststrategie

**T5.1** Unit Tests für KillSwitch (activate/deactivate/is_active, thread-safety).

**T5.2** Unit Tests für RateLimiter (global + per-symbol limits, concurrency).

**T5.3** Unit Tests für OrderDeduplicator (dup detection, thread-safety).

**T5.4** Unit Tests für ExecutionService (full flow: validate → dedupe → rate → execute).

**T5.5** Integrationstest: ExecutionService + KillSwitch + RateLimiter zusammen.

**T5.6** Negative Tests: Kill Switch active, Rate Limit exceeded, Duplicate orders.

**T5.7** Security Tests: Credentials never logged, live_enabled=false blocks execution.

**T5.8** Regressionstests: Bestehende Phasen 1-4 müssen weiterhin funktionieren.

## 9. Definition of Done

- Alle Anforderungen R5.1–R5.24 implementiert
- make check clean (Tests, Linting, Type Checking)
- docs/handoff.md aktualisiert
- Live Execution standardmäßig deaktiviert
- Keine spezifische Exchange-Integration (nur Interface)