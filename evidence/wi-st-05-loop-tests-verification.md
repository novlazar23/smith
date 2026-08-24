# WI-ST-05 Verification Evidence — ShadowTradingLoop-Tests

Datum: 2026-08-24 · Workitem: WI-ST-05 · Reviewer: local-critic Oracle (approved)

## Testlauf (final)
```
uv run pytest -q tests/test_shadow_trading_loop.py
27 passed in 1.81s
```

## Full Gate (make check)
```
884 passed, 1 warning in 60.64s
uv run ruff check src tests
All checks passed!
uv run mypy src
Success: no issues found in 55 source files
```

## Abgedeckte Spec-Items (12 neue Chunk-3-Tests + 15 bestehende)
- ST.6/E2: NO_TRADE-/Risk-Reject-Record-Semantik (NO_ACTIVE_CHAMPIONS,
  BELOW_MIN_CONFIDENCE, SYMBOL_NOT_ALLOWED)
- ST.9/ST.10: Stop-Loss/Target-Closure innerhalb derselben Iteration; M2M läuft ohne
  Budget weiter
- ST.11/S2: Kill-Switch- und Flags-Self-Stops innerhalb einer Iteration (kein M2M bei
  Kill)
- ST.12/O4: Phase-2-Metriken (directional_accuracy 0.75, Brier 0.25, expectancy −2.0,
  max_dd 0.12 über Records [LONG/+10, SHORT/−5, SHORT/−7, LONG/0])
- O1/ST.13: strukturierte INFO-Logs + vollständige Audit-Kette (Run-Lifecycle +
  SHADOW_ITERATION unter run_id; SHADOW_DECISION unter decision_id; SHADOW_FILL unter
  trade_id — get_audit_log(entity_id)-Filter respektiert)
- ST.14: Determinismus bei fixen Inputs; State-Größenbegrenzung (monkeypatch
  MAX_RECORDS/MAX_PORTFOLIO_HISTORY/_RECENT_PORTFOLIO_EXACT)
- F6/S3: Consecutive-Error-Self-Stop; fail-closed INCOMPLETE_FILL_DATA → REJECTED-Record
  mit unverbrauchtem Preliminary-Sizing (250 @ 100)
- ST.4: Budget-Erschöpfung bleibt RUNNING, Audit-Ankündigung genau einmal pro UTC-Tag;
  NO_TRADE konsumiert kein Budget

## Behobene Defekte während Verifikation
1. Loop schrieb `symbols` nie in den Session-State (nur `start()`) → `"symbols": symbols`
   im finalen update_state von `run_once()` (ST.8).
2. Audit-Lookups: SHADOW_DECISION/SHADOW_FILL hängen an decision_id/trade_id, nicht an
   run_id → Tests nutzen globalen Log für diese Einträge.
3. Spy-Test: `autospec=True` liefert auf diesem Python-Build (3.12.3) nackte Funktionen
   ohne `.stop()` → Patch-Objekte werden gestoppt, kein autospec.
4. FakeAgentRuntime wiederholt den letzten Script-Eintrag → explizites `reset_script`
   vor der dritten Iteration im NO_TRADE-Budget-Test.
5. Session-Default-Status ist STOPPED (models.py L478) → RUNNING-Precondition im
   M2M-Budget-Test gesetzt (wie in den bestehenden Budget-Tests).
6. Ruff RUF007: `zip(ticks, ticks[1:])` → `itertools.pairwise(ticks)`.

## Isolation verifiziert
Live-Endpoints (`submit_order` auf LiveExecutionService + Binance/Bitget/Bybit/Coinbase,
`get_order_status`, `cancel_order`) sind bei FILLED-Trades nie gerufen — nur Paper-Stack
(`trade_store` enthält genau 1 Trade).

## Unabhängiges Review (local-critic Oracle): APPROVED
- „no path from shadow loop to live execution“ bestätigt.
- Spec-Coverage ST.1–ST.14/F5/F6/O1/O4/S2/Z2 je mit Testnamen verifiziert.
- Nicht blockierende Findings (Folge-Workitem-Kandidaten):
  (MINOR) M2M-Ticker-Fetch auch für Positionen außerhalb der konfigurierten Symbole;
  (NIT) Determinismus-Test könnte vollständigen Audit-Log-Vergleich ergänzen.
