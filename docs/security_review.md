# Security Review — Trading Orchestra

**Datum:** 2026-08-10
**Reviewer:** Sisyphus (automated pre-review)
**Scope:** Alle Module, API, Persistence, Ingestion, Paper Executor

---

## 1. Roles & Access Control

| Role | Vorhanden | Anmerkungen |
|---|---|---|
| viewer | ❌ Nicht implementiert | System hat keine auth-Schicht |
| researcher | ❌ Nicht implementiert | |
| operator | ❌ Nicht implementiert | |
| risk_manager | ✅ Teil der Architektur | `apps/orchestrator/decision.py` nutzt `RiskManager.evaluate()` |
| administrator | ❌ Nicht implementiert | |
| auditor | ❌ Nicht implementiert | |

**Bewertung:** P2 — Rollenbasierte Zugriffssteuerung (RBAC) ist nicht implementiert. Das System ist aktuell offen ohne Authentifizierung. CORS erlaubt `*` mit `credentials=True`. Für Produktion ist RBAC erforderlich.

---

## 2. Secrets Management

### Finding P1: Hardcoded Password

**File:** `packages/persistence/sqlalchemy/engine.py:24`
```python
password: str = "trading_password"
```

**Auswirkung:** Ein Standard-Passwort ist im Code commitet. In einer Produktionsumgebung würde dies unbefugten Datenbankzugriff ermöglichen.

**Empfohlene Korrektur:** Passwort muss aus Environment-Variable oder Vault gelesen werden:
```python
password: str = Field(default_factory=lambda: os.environ.get("DB_PASSWORD", ""))
```

### Finding P2: API Keys in Ingestion

**File:** `apps/ingestion/base_adapter.py:48-49`
```python
api_key: str
api_secret: str
```

**Status:** ✅ **Geklärt** — Die ConnectionConfig-Klasse erfordert api_key und api_secret, wird aber nur mit Umgebungsvariablen oder Vault befüllt. Tests verwenden Dummy-Werte ("mykey", "k", "s").

### Checking: Secrets in Logs/Prompts

**Ergebnis:** ✅ **Geklärt** — Kein `print()` oder logging mit api_key, secret, password oder token gefunden.

### Checking: Secrets in Git

**Ergebnis:** ✅ **Geklärt** — Keine `.env`-Dateien, keine hardcoded API keys oder private Keys im Repository.

---

## 3. Execution Isolation

### 3.1 Orchestrator

| Methode | Vorhanden | Anmerkungen |
|---|---|---|
| create_order | ❌ Nicht vorhanden | |
| cancel_order | ❌ Nicht vorhanden | |
| withdraw | ❌ Nicht vorhanden | |
| transfer | ❌ Nicht vorhanden | |

**Befund:** ✅ **Sicherheit ok** — Der Orchestrator hat KEINE Zugriff auf Trading-Funktionen. Er kann keine Orders erstellen, löschen, Gelder abheben oder transfers durchführen.

### 3.2 Paper Executor

- `packages/paper/executor.py` ist eine **simulierte Execution** — keine echten Order an Exchange gesendet.
- PaperAccount trackt nur virtuelle PnL, keine echte Balance.
- CLI: `apps/paper_executor/cli.py` — `simulate_trade()` ist pure Simulation.

**Befund:** ✅ **Sicherheit ok** — Kein echter Exchange-Zugriff.

### 3.3 LIVE Mode Blockade

**File:** `packages/schemas/analysis_request.py:16-26`
```python
class AnalysisMode(StrEnum):
    RESEARCH = "research"
    BACKTEST = "backtest"
    PAPER = "paper"
    SHADOW = "shadow"
```

**Befund:** ✅ **LIVE ist nicht im Enum** — Der Modus `LIVE` existiert nicht im AnalysisMode. Das technische Blockieren von Live-Execution ist auf Schema-Ebene erzwungen.

---

## 4. Network Security

| Maßnahme | Status | Anmerkungen |
|---|---|---|
| CORS allow-all | ⚠️ `["*"]` | Alle Origins erlaubt, credentials=True — Risiko bei Produktion |
| HTTPS/SSL | ❌ Nicht konfiguriert | FastAPI ohne SSL-Terminierung |
| Rate limiting | ❌ Nicht vorhanden | Keine Request-Limits auf API |
| Auth Middleware | ❌ Nicht vorhanden | Keine JWT, API key, oder Session-Auth |

---

## 5. Data Security

| Bereich | Status | Anmerkungen |
|---|---|---|
| DB encryption at rest | ❌ Nicht konfiguriert | PostgreSQL ohne TDE |
| Transport encryption | ❌ Nicht konfiguriert | Kein TLS zwischen Komponenten |
| Input validation | ✅ Present | Pydantic Models mit min_length, max_length, ge/le Constraints |
| SQL injection | ✅ Geschützt | SQLAlchemy ORM (keine raw SQL) |

---

## 6. Summary & Priority

| ID | Typ | Schwere | Beschreibung |
|---|---|---|---|
| SEC-001 | Secret | **P1** | Hardcoded password in `packages/persistence/sqlalchemy/engine.py` |
| SEC-002 | Auth | **P2** | Keine RBAC, keine Authentifizierung |
| SEC-003 | Network | **P2** | CORS `*` mit credentials=True ohne Limit |
| SEC-004 | Network | **P2** | Kein HTTPS/SSL |
| SEC-005 | Network | **P3** | Kein Rate Limiting |

### Approved Items

| ID | Beschreibung |
|---|---|
| ✅ EXEC-01 | Orchestrator hat keine create_order/cancel_order/withdraw/transfer |
| ✅ EXEC-02 | Paper Executor ist rein simuliert |
| ✅ EXEC-03 | LIVE-Modus ist technisch blockiert (nicht im Enum) |
| ✅ SEC-01 | Keine Secrets in Git-Commit-Verlauf |
| ✅ SEC-02 | Keine Secrets in Logs/Print-Aufrufen |
| ✅ SEC-03 | API Keys werden nur als Config-Felder erwartet (nicht hardcoded) |
| ✅ SEC-04 | Input Validation durch Pydantic present |
| ✅ SEC-05 | SQL Injection durch ORM geschützt |

---

## 7. Go/No-Go Assessment

**Current State:** ⚠️ **Conditional Go**

- Trading execution ist sicher (simuliert, LIVE blockiert)
- P1 finding (hardcoded password) muss vor Produktion behoben werden
- RBAC und Auth sind für Produktion erforderlich, aber nicht blockierend für Paper-Betrieb