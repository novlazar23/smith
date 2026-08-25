# Architecture Notes

The initial implementation intentionally uses in-memory stores for agents and snapshots.
The next milestone is PostgreSQL persistence plus immutable run/audit records.

Live execution remains out of scope until shadow and paper evaluation are stable.
The future execution component should be deployed as a separate service with isolated credentials.

## ADR: Isolation des Shadow-Trading-Loops (Shadow-Trading-Epic, WI-ST-01 bis WI-ST-06)

Der Shadow-Trading-Loop (`services/shadow_trading_loop.py`) ist strikt von Live-Execution
getrennt. Die Isolation ist ein Entwurfsprinzip und wird zusätzlich durch Negativtests
erzwungen:

- `MarketDataProvider` (neues strikt read-only Protokoll in `shadow_trading_loop.py`):
  genau eine Methode `get_ticker(symbol: str) -> dict[str, float]` (Feld `last`), keine
  Trade- oder Write-Methoden. Der Produktions-Provider (`CryptoMarketDataProvider`)
  bedient sich ausschließlich aus dem bestehenden Crypto-Marktdaten-Router
  (`CryptoExecutionRouter.get_ticker`).
- `ShadowExecutionBackend` (`services/shadow_execution_backend.py`): dünner Wrapper
  ausschließlich um den bestehenden `PaperExecutionStack`; keine eigene
  Simulationslogik, kein Pfad zu Live-Adaptern, deterministisches Status-Mapping
  (FILLED/REJECTED/ERROR, abweichende Adapter-Statuses fail-closed als ERROR).
- Enforcement durch Negativtests: Spy-Tests asserten 0 Aufrufe aller
  Exchange-/Live-Write-Pfade (`tests/test_shadow_execution_backend.py`:
  `test_execute_never_reaches_exchange_write_endpoints`; `tests/test_shadow_trading_loop.py`:
  Live-Endpoint-Isolation über Spies auf den Crypto-Adaptern und
  `LiveExecutionService`), sowie ein Modul-Import-Test
  (`test_backend_module_imports_no_exchange_or_live_execution_paths`), der per AST
  prüft, dass das Backend-Modul keine Exchange- oder Live-Execution-Module importiert.
- LLM-Ausgaben liefern nur Vorschläge (Trade-Proposals); vor jeder Decision greift die
  deterministische Risk Engine erneut (Risk Re-Check, inkl. Kill-Switch-Status). Der
  Loop respektiert den globalen Kill Switch und stoppt sich selbst, sobald dieser
  aktiv ist.
