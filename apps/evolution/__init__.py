"""Evolutions-Pipeline: autonome Strategie-Entwicklung mit Data-Snooping-Firewall.

Der Orchestrator (15-min-Zyklus) läuft unverändert weiter. Dieser App-Modus
führt **nächtliche Evolutions-Zyklen** aus:

1. **Auswerten** (immer an): State + Live-Paper (Demo-Trader) → Evidenz-Digest.
2. **Diskutieren** (LLM, opt-in): Personas schlagen 1-3 Hypothesen vor.
3. **Preregistrieren**: Hypothese wird VOR dem Test vollständig fixiert
   (Claim, Variante, Testplan, Entscheidungsregel) — ``models.py``.
4. **Umsetzen**: Konfig-Variante (existierende Strategie) oder neuer
   Mechanismus (Code im Jail, ``mechanism.py``).
5. **Testen** (deterministisch): Backtest-Matrix Kalibrierung + OOS über
   Assets, 5m, Flatsize, de-fakti-Kosten — ``evaluate.py``.
6. **Beurteilen** (deterministisch, kein LLM): mechanische
   Preregistrierungs-Regel → Registry (promoted) oder Grab (rejected) —
   ``evaluate.judge``.
7. **Promoten**: Promotions nur in die Paper-Ebene (``live_trading_enabled``
   bleibt aus); neue Mechanismen-Strategien werden exportiert, Adoption in
   den Host-Tree ist eine bewusste Host-Entscheidung (Gates: pyright/ruff/pytest).

State: JSONL-Dateien (``state.py``) im State-Dir (Container:
``/app/backtest_reports/evolution`` via bestehendem Volume; Host: ``./evolution``
via Nightly-Export).
"""
