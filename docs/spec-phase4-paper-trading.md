# Phase 4 — Paper Trading

## 1. Problem

Das System hat Research Runtime (Phase 1), Evaluation (Phase 2) und Evolution (Phase 3)
implementiert. Die Trading-Entscheidungen bleiben jedoch abstrakt — sie werden im
`TradingRun` gespeichert, aber nie ausgeführt.

Es fehlt eine **simulierte Trading-Engine**, die:
- Order simulation mit realistischen Annahmen (Slippage, Queue-Füllungsrate)
- Position lifecycle management (Open, Hold, Close)
- Portfolio State tracking (aggregierter PnL, exposure, risk metrics)
- Performance attribution (agent contribution to PnL)
- Deterministische Integration in den bestehenden Risk-Engine-Flow

## 2. Anforderungen

### 2.1 Paper Exchange

**R4.1** Muss Orders gegen einen simulierten Mid-Preis ausführen (nicht gegen den Close-Preis
des Snapshots, um Look-Ahead zu vermeiden).

**R4.2** Muss Slippage deterministisch berechnen basierend auf `expected_slippage_bps` aus
`TradeProposal` und Policy-Konfiguration.

**R4.3** Muss Order-Filling-Rate konfigurierbar haben (Standard: 80% für Limit-Orders).

**R4.4** Muss Order-Status tracken: `PENDING` → `FILLED` / `PARTIALLY_FILLED` / `REJECTED`.

### 2.2 Position Lifecycle

**R4.5** Jede ausgeführte Order erzeugt eine `PaperPosition` mit:
- `position_id` (eindeutig)
- `symbol`, `side` (LONG/SHORT)
- `entry_price`, `quantity`, `fees`
- `entry_timestamp`
- Status: `OPEN` / `CLOSED` / `CANCELLED`

**R4.6** Muss Partial-Closes unterstützen (z.B. 50% der Position schließen).

**R4.7** Muss Stop-Loss und Take-Profit automatisch auslösen.

**R4.8** Muss PnL berechnen bei Schließung:
- LONG: `(exit_price - entry_price) * quantity - fees`
- SHORT: `(entry_price - exit_price) * quantity - fees`

### 2.3 Portfolio State

**R4.9** Muss aggregierte Portfolio-Metriken tracken:
- Gesamtes Equity (start + unrealized + realized PnL)
- Pro-Position unrealized PnL (Mark-to-Market)
- Total realized PnL
- Drawdown (max, current)
- Pro-Symbol exposure

**R4.10** Muss Portfolio-State bei jedem Snapshot-Update aktualisieren (Mark-to-Market).

### 2.4 Performance Attribution

**R4.11** Muss Performance Records pro Agent erstellen basierend auf Paper-Trading-Ergebnissen.

**R4.12** Muss Agent-spezifische Metriken tracken:
- Anzahl Trades
- Win Rate
- Avg PnL per Trade
- Sharpe Ratio (täglich)
- Max Drawdown

### 2.5 Datenbank-Persistenz

**R4.13** Muss alle Paper-Trades und Positions in PostgreSQL persistent speichern.

**R4.14** Muss Migration für neue Tabellen bereitstellen:
- `paper_trades` — Order executions
- `paper_positions` — Positions lifecycle
- `portfolio_state` — aggregierte Portfolio-Metriken

**R4.15** Muss In-Memory-Fallback haben, wenn PostgreSQL nicht verfügbar ist.

### 2.6 API-Routen

**R4.16** Muss API-Endpunkte bereitstellen:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/paper/trades` | Submit order (after risk approval) |
| GET | `/paper/trades` | List all paper trades |
| GET | `/paper/trades/{trade_id}` | Get trade details |
| GET | `/paper/trades/symbol/{symbol}` | Filter by symbol |
| POST | `/paper/positions/{position_id}/close` | Close position manually |
| POST | `/paper/positions/{position_id}/partial` | Partial close |
| GET | `/paper/positions` | List all positions |
| GET | `/paper/positions/open` | List open positions |
| GET | `/paper/portfolio` | Get current portfolio state |
| GET | `/paper/portfolio/history/{run_id}` | Historical portfolio for run |
| POST | `/paper/run/{run_id}/execute` | Execute run decisions in paper mode |
| GET | `/paper/performance/agent/{agent_id}` | Agent performance summary |

### 2.7 Integration

**R4.17** Muss in den bestehenden `TradingRun`-Flow integrieren:
- Nach `DECISION` → `POST /paper/run/{run_id}/execute` startet Paper-Execution
- Erstellt Paper-Orders für alle risk-approved Entscheidungen
- Updates PerformanceRecords mit realisierten PnL-Werten

**R4.18** Muss `ExecutionGateway` erweitern statt ersetzen:
- `ExecutionGateway` behält `LIVE_EXECUTION_DISABLED` für Safety
- `PaperExecutionService` ist eine neue, separierte Komponente
- Keine Änderungen an bestehender `ExecutionGateway.submit()`-Logik

## 3. Architektur

```
┌─────────────────────────────────────────────────────┐
│              Paper Trading Flow                      │
│                                                      │
│  TradingRun (DECISION)                               │
│       │                                              │
│       ▼                                              │
│  PaperExecutionService                               │
│       │                                              │
│       ├──► PaperExchange (order matching)            │
│       │       │                                      │
│       │       ├──► Slippage Model                   │
│       │       ├──► Fill Rate Model                  │
│       │       └──► Fee Model (0.1% per trade)       │
│       │                                              │
│       ├──► PositionManager (lifecycle)               │
│       │       │                                      │
│       │       ├──► Stop-Loss / Take-Profit           │
│       │       ├──► Partial Closes                    │
│       │       └──► PnL Calculation                   │
│       │                                              │
│       └──► PortfolioTracker (aggregate state)        │
│               │                                      │
│               ├──► Equity Curve                      │
│               ├──► Drawdown                          │
│               └──► Exposure Tracking                 │
│                                                      │
├─────────────────────────────────────────────────────┤
│              Data Layer                              │
│                                                      │
│  PostgreSQL:                                         │
│  ┌──────────────────────────────────────────────┐   │
│  │ paper_trades (id, trade_id, run_id, ...)     │   │
│  │ paper_positions (id, trade_id, status, ...)  │   │
│  │ portfolio_state (run_id, equity, pnl, ...)   │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  In-Memory Fallback:                                 │
│  ┌──────────────────────────────────────────────┐   │
│  │ InMemoryPaperTradeStore                      │   │
│  │ InMemoryPositionStore                        │   │
│  │ InMemoryPortfolioStore                       │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## 4. Datenmodelle

### 4.1 PaperTrade (extends TradeProposal)

```python
class PaperTradeStatus(StrEnum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

class PaperTrade(BaseModel):
    id: str = Field(default_factory=lambda: f"paper-trade-{uuid4()}")
    trade_id: str  # Original decision_id from TradingRun
    run_id: str
    symbol: str
    side: str  # LONG/SHORT
    equity: float
    entry_price: float
    requested_leverage: float
    requested_quantity: float  # calculated from equity, leverage, stop_distance
    actual_quantity: float  # after fill rate
    actual_price: float  # after slippage
    stop_price: float
    target_price: float
    fill_rate: float  # 0.0-1.0
    slippage_bps: float
    fees: float = 0.0
    status: PaperTradeStatus = PaperTradeStatus.PENDING
    partial_fills: list[dict] = Field(default_factory=list)  # [{price, qty, timestamp}]
    created_at: datetime = Field(default_factory=utcnow)
    filled_at: datetime | None = None
    closed_at: datetime | None = None
```

### 4.2 PaperPosition

```python
class PaperPositionStatus(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    STOPPED_OUT = "STOPPED_OUT"
    TARGET_HIT = "TARGET_HIT"

class PaperPosition(BaseModel):
    id: str = Field(default_factory=lambda: f"paper-pos-{uuid4()}")
    trade_id: str
    run_id: str
    symbol: str
    side: str  # LONG/SHORT
    entry_price: float
    quantity: float
    fees: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    status: PaperPositionStatus = PaperPositionStatus.OPEN
    open_timestamp: datetime = Field(default_factory=utcnow)
    close_timestamp: datetime | None = None
    close_price: float | None = None
    close_reason: str | None = None  # MANUAL, STOP_LOSS, TARGET_HIT
```

### 4.3 PortfolioState

```python
class PortfolioState(BaseModel):
    run_id: str
    start_equity: float = 100000.0  # configurable default
    current_equity: float = 100000.0
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    peak_equity: float = 100000.0
    positions: dict[str, float] = Field(default_factory=dict)  # symbol -> quantity
    symbols: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=utcnow)
```

## 5. Algorithmen

### 5.1 Slippage Model

```python
# Deterministisch basierend auf slippage_bps
slippage = abs(entry_price * slippage_bps / 10000)

if side == "LONG":
    actual_price = entry_price + slippage
else:  # SHORT
    actual_price = entry_price - slippage
```

### 5.2 Fill Rate Model

```python
# Deterministisch: fill_rate aus Policy, mit +/- 5% Jitter
import random
jitter = random.uniform(-0.05, 0.05)
actual_fill_rate = max(0.0, min(1.0, fill_rate * (1 + jitter)))
```

### 5.3 PnL Calculation

```python
if side == "LONG":
    pnl = (exit_price - entry_price) * quantity - fees
else:  # SHORT
    pnl = (entry_price - exit_price) * quantity - fees
```

### 5.4 Stop-Loss / Take-Profit Trigger

```python
# Bei jedem Snapshot-Update prüfen
if position.status == PaperPositionStatus.OPEN:
    if side == "LONG":
        if current_price <= position.stop_price:
            # Stop-Loss ausgelöst
            close_price = position.stop_price
            close_reason = "STOP_LOSS"
        elif position.target_price and current_price >= position.target_price:
            # Target getroffen
            close_price = position.target_price
            close_reason = "TARGET_HIT"
    elif side == "SHORT":
        if current_price >= position.stop_price:
            close_price = position.stop_price
            close_reason = "STOP_LOSS"
        elif position.target_price and current_price <= position.target_price:
            close_price = position.target_price
            close_reason = "TARGET_HIT"
```

## 6. Fehlerbehandlung

**F4.1** Ungültige Order (z.B. negative quantity) → REJECTED, Audit-Log-Eintrag.

**F4.2** Risk-Engine-Ablehnung nach Paper-Submission → Order wird nicht erstellt.

**F4.3** PostgreSQL-Verbindungsausfall → In-Memory-Fallback, keine Datenverluste.

**F4.4** Duplicate trade_id → REJECTED mit "DUPLICATE_DECISION_ID".

**F4.5** Ungültiger Symbol-Name → REJECTED mit "SYMBOL_NOT_ALLOWED".

## 7. Security

**S4.1** Paper Trading muss durch Policy konfiguriert werden (`paper_trading_enabled: true`).

**S4.2** Keine echten Gelder oder API-Credentials involviert.

**S4.3** Audit-Log für alle Paper-Trades und Position-Änderungen.

## 8. Teststrategie

**T4.1** Unit Tests für PaperExchange (Slippage, Fill Rate, Fees).

**T4.2** Unit Tests für PositionManager (Open, Close, Partial Close, Stop-Loss, Target).

**T4.3** Unit Tests für PortfolioTracker (Equity Curve, Drawdown, Exposure).

**T4.4** Integrationstest: Full Paper-Execution-Flow von TradeProposal → Order → Position → PnL.

**T4.5** Datenbank-Tests: Persistenz und Recovery bei PostgreSQL-Ausfall.

**T4.6** Negative Tests: Ungültige Orders, Duplicate trade_ids, unzulässige Symbole.

**T4.7** Regressionstests: Bestehende Phasen 1-3 müssen weiterhin funktionieren.

## 9. Akzeptanzkriterien

1. [ ] PaperTrade wird erstellt, Slippage berechnet, Fill Rate angewendet
2. [ ] PaperPosition wird bei FILLED erstellt, bei STOP_LOSS/TARGET_HIT geschlossen
3. [ ] PortfolioState wird bei jedem Snapshot-Update aktualisiert
4. [ ] API-Endpunkte für Trades, Positions, Portfolio funktionieren
5. [ ] PostgreSQL-Persistenz mit In-Memory-Fallback
6. [ ] Integration mit TradingRun-Flow: `POST /paper/run/{id}/execute`
7. [ ] 30+ Tests, make check clean
8. [ ] docs/handoff.md aktualisiert
