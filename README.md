# Trading Orchestra

## Running the system

Der komplette Stack startet mit einem Befehl — Schemas, Redpanda-Topic und
alle Services werden automatisch gebootstrapped:

```bash
docker compose up -d --build
```

`db-init` (One-Shot) legt bei jedem Start idempotent an:
PostgreSQL-Tabellen (`trading`), ClickHouse-Tabellen (`trading_events`),
Redpanda-Topic `market_data`. Die App-Services starten erst danach.

### Laufende Services

| Service | Aufgabe | Persistiert in |
|---|---|---|
| `api` | REST-API auf `localhost:8080` (`/status`, `/metrics`, `/v1/...`) + zentrale Web-UI unter `http://localhost:8080/` (Tabs: Trading, Monitoring, Metriken, Alerts, ML, Storage) | — |
| `market-producer` | Echte Binance-Futures-Klines (BTC/ETH, 60 s Ticks, 1 m), Dummy-Fallback pro Tick bei Ausfall | Redpanda `market_data` |
| `ingestion-consumer` | Konsument von `market_data`, validiert Candles | ClickHouse `trading_events.candles` |
| `news-ingestion` | RSS-Zyklen (30 s), dedupliziert + klassifiziert | PostgreSQL `news_events` |
| `orchestrator` | Shadow-Pipeline im 15-Min-Zyklus (Agenten → Konsens, **keine Order-Ausführung**) | PostgreSQL `shadow_decisions` |
| `demo-trader` | Paper-Trading im 5-Min-Zyklus auf echten Kursdaten (ACTIVE-Agenten → Konsens → **virtuelle** Orders, 100.000 $ Startkapital) | PostgreSQL `demo_trades` + `demo_account` |
| `alertmanager` | Alert-Ziel von Prometheus auf `127.0.0.1:9093` | — |

Dazu: `postgres`, `clickhouse`, `redis`, `minio`, `redpanda`, `mlflow`,
`prometheus` (`127.0.0.1:9090`), `grafana` (`127.0.0.1:3000`).

### Verifikation

```bash
curl -s localhost:8080/status | python3 -m json.tool   # "status": "running"
curl -s localhost:8080/metrics | grep -m1 http_requests_total
docker compose exec -T postgres psql -U orchestra -d trading -tc "SELECT count(*) FROM news_events"
curl -s localhost:9093/api/v2/status | grep -o '"status":"ready"'
curl -s localhost:9090/api/v1/rules | grep -o 'trading-orchestra-alerts'
```

CH-Candle-Zahl (wächst pro Producer-Tick):

```bash
PW=$(cat configs/secrets/clickhouse_password.txt)
docker compose exec -T clickhouse wget -qO- \
  --header="Authorization: Basic $(printf 'orchestra:%s' "$PW" | base64 -w0)" \
  --header="X-ClickHouse-Database: trading_events" \
  --post-data "SELECT count() FROM candles" http://127.0.0.1:8123/
```

### Backtest & Kalibrierung (One-Shot)

Die Agenten-Ensembles sind regel-/TA-basiert (keine ML-Gewichte) — sie
werden daher nicht „trainiert", sondern auf historischen Kursdaten
**geprüft und kalibriert**: exakt derselbe Produktions-Pfad wie im
Live-Betrieb (ACTIVE-Ensemble → OrchestratorPipeline → Konsens →
Konfidenz-Gate), nur rückwärts auf Kerzen-Historik.

1. **`backfill`** lädt historische Binance-Futures-1m-Kerzen idempotent in
   ClickHouse `trading_events.candles_history` (gleiche Spalten wie die
   Live-Tabelle `candles`, aber **ohne TTL** — die Live-Tabelle löst Daten
   nach 1 Jahr auf, die dedizierte Backtest-Tabelle nicht; sie wird
   automatisch angelegt, vorhandene Zeiträume werden übersprungen,
   Lücken nachgeladen):

   ```bash
   docker compose --profile on-demand run --rm backfill \
     python -m apps.backfill --start 2021-05-01 --end 2022-07-31
   docker compose --profile on-demand run --rm backfill \
     python -m apps.backfill --months 12
   ```

2. **`backtest`** führt die Produktions-Entscheidungslogik auf den
   historischen Kerzen aus — pro Marktregime ein Szenario (`crash-2021-05`,
   `pump-2021-11`, `crash-2022-06`, `range-2022-03`, `full` = gesamte
   vorhandene Historie) — und erzeugt pro Szenario einen MLflow-Run
   (Experiment `backtest`, sichtbar im ML-Tab der Web-UI), einen
   Markdown-Report auf der Konsole und Artefakte in `./backtest_reports/`
   (`report.json`, `equity_curve.csv`, `evaluations.json`):

   ```bash
   docker compose --profile on-demand run --rm backtest \
     python -m apps.backtest \
     --scenarios crash-2021-05,pump-2021-11,crash-2022-06,range-2022-03 \
     --gate 0.3 --sweep-gates 0.2,0.3,0.4,0.5,0.6,0.7
   ```

   Der Gate-Sweep rechnet die gecachten Konsens-Entscheidungen des letzten
   Szenarios pro Gate nach (keine Agenten-Rekomputation) und liefert die
   Kalibrierungstabelle (Return, Sharpe, Max-Drawdown, Win-Rate pro
   Gate) — die Evidenz-Basis dafür, ob das Konfidenz-Gate (Default 0,3)
   und die Agenten-Setups passen. `--resample 5m` aggregiert die 1m-Kerzen
   auf 5m (4× weniger Evaluations, gröberes Fenster).

   Zusätzlich zu `--gate`/`--sweep-gates` (nur Agenten-Ensemble) gibt es
   zwei weitere Strategie-Modi (mutuell exklusiv):

   - **Bibliothek-Strategien** (`packages/strategies/`, deterministisch,
     ohne LLM): `--list-strategies` listet alle Strategien mit Parametern;
     `--strategy NAME` backtestet eine einzelne Strategie; `--params
     fast=8,slow=30` überschreibt Parameter; `--sweep-library` läuft alle
     Strategien × alle Szenarien durch (je Run ein Artefakt-Verzeichnis
     `szenario__strategie`):

     ```bash
     docker compose --profile on-demand run --rm backtest \
       python -m apps.backtest --list-strategies
     docker compose --profile on-demand run --rm backtest \
       python -m apps.backtest --scenarios crash-2021-05,pump-2021-11,crash-2022-06,range-2022-03 \
       --resample 5m --sweep-library
     ```

   - **LLM-Prompt-Strategie** (`PromptStrategy`, OpenAI-kompatibler
     Endpoint via `litellm`, Default `local-fast`): Der LLM sieht pro
     `--llm-every`-ter Kerze einen deterministischen OHLCV-Snapshot
     (zuletzt 16 Kerzen, EMA/RSI/ATR/MACD/VWAP, Position, Kosten,
     Exit-Regeln) und liefert eine strukturierte Entscheidung
     (`BUY/SELL/CLOSE` + Konfidenz + Stop/Take-Profit). `--llm-cache
     DATEI` speichert Antworten als JSONL (wiederverwendbar, determini-
     stischer Re-Run), `--llm-model` überschreibt das Modell,
     `--min-candles` ist das Warmup (Default 30, für Prompt-Läufe ≥ 120
     sinnvoll):

     ```bash
     docker compose --profile on-demand run --rm backtest \
       python -m apps.backtest --scenario crash-2021-05 --resample 5m \
       --prompt-strategy --llm-every 60 --min-candles 120 \
       --llm-cache /app/backtest_reports/llm_cache.jsonl
     ```

    **Erster Kalibrierungslauf (02.09.2026, BTC/USDT, 2021-05→2022-07,
    157.777 Kerzen):** In allen vier Szenarien (crash-2021-05, pump-2021-11,
    crash-2022-06, range-2022-03) fällt der Konsens zu 100 % auf `NO_TRADE`
    aus (18.124 Evaluations), der Gate-Sweep (0,2–0,7) ändert daran nichts.
    Die Konfidenz liegt im Median bei 0,26–0,28 und übersteigt das Gate 0,3
    nur in 15–22 % der Fälle — das Gate ist also **nicht** der Engpass.
    Ursache: Bei drei gleichgewichtigten Agenten kann im `compute_consensus`
    keine Seite die notwendige Gewichtsmehrheit (50 %) erreichen, solange
    nicht mindestens zwei Agenten in dieselbe Richtung voten; die aktuelle
    Vot-Logik (`up_score > 0,6`) ist so streng, dass das in den historischen
    Fenstern nie passiert. Hebel für einen Folge-Lauf:
    `min_consensus_threshold` auf 0,25 setzen, die 0,6-Vote-Schwelle der
    Agenten senken oder ein viertes, unabhängiges Signal-Agent-Setup
    ergänzen. Erst danach ist eine Gate-Kalibrierung sinnvoll.

    **Zweiter Kalibrierungslauf (02.09.2026, neues 4er-Ensemble):** Die
    drei alten Agenten (Anomaly, Historical-Analogy, Chart) wurden durch
    vier richtungssensitive ersetzt, die unterschiedliche Marktperspektiven
    abbilden: ``trend`` (EMA12/26-Ausrichtung, ROC10, ATR-normierte
    Trennung), ``mean_reversion`` (z-Score gegen SMA50, RSI14),
    ``volatility_regime`` (Squeeze-Breakout aus Baseline-Bandbreite plus
    20-Bar-Position) und ``volume_conviction`` (Up/Down-Volumen-Verhältnis,
    OBV-Steigung, Partizipation). Parallel wurde
    ``min_consensus_threshold`` von 0,5 auf 0,2 gesenkt: Bei vier
    gleichgewichtigten Agenten reicht eine Richtungsvote (0,25 > 0,2) für
    eine Konsens-Entscheidung, das Konfidenz-Gate 0,3 erzwingt in der
    Praxis zwei übereinstimmende Agenten (Konfidenz ≈ 0,5).

    Ergebnis (Startkapital 100.000 $, Long-Only, max. 10 % Position):

    | Szenario | Return | Trades | Win-Rate | Gate-Pass ≥ 0,3 |
    |---|---:|---:|---:|---:|
    | crash-2021-05 (11 Tage) | -4,4 % | 1 | 0 % | 98,6 % |
    | pump-2021-11 (9 Tage) | -4,2 % | 1 | 0 % | 98,7 % |
    | crash-2022-06 (10 Tage) | -5,0 % | 1 | 0 % | 99,2 % |
    | range-2022-03 (31 Tage) | -13,2 % | 1 | 0 % | 99,1 % |

    Der strukturelle `NO_TRADE`-Blockade aus dem ersten Lauf ist damit
    gelöst: Die Agenten votieren in 15–40 % der Evaluations richtungs-
    bestimmt, und der Konsens übersteigt das Gate 0,3 in 98,6–99,2 % aller
    Evaluations. Die Handelsqualität ist dagegen negativ: alle vier
    Szenarien laufen ins Minus. Der Gate-Sweep auf crash-2021-05 liefert
    kein positives Gate (0,2: 934 Trades/38,1 % Win, 0,3–0,5: 899/36,6 %,
    0,6: 255/40,9 %, alle negative Return) — der Engpass liegt nicht im
    Gate, sondern in der Entry-/Exit-Logik.

    **Korrektur (im Zuge des dritten Laufs gefunden):** Die Backtest-Engine
    bewertete Positionen zur Kostengrundlage (`qty × avg_price`) statt
    mark-to-market und stellte Glattstellungssignale ebenfalls zur
    Kostengrundlage statt zum Marktpreis; geschlossene Positionen verschwanden
    aus dem Positions-Dict, sodass `total_trades` pro Symbol auf 1 kollabierte.
    Die „Returns" der Lauf-2-Tabelle sind daher weitgehend **Kostenabrieb**
    (Slippage/Commission über hundert Round-Trips), keine Markt-PnL, und die
    Spalten „Trades"/„Win-Rate" waren Artefakte. Die Confidence-Bucket-
    Statistik (close-preis-basierte Rekonstruktion) war dagegen schon im
    zweiten Lauf die realitätsnahe Sicht: ~38 % Win-Rate mit negativem
    mittleren PnL pro Trade. Die Engine wurde im dritten Lauf korrigiert
    (mark-to-market, Marktpreis-Glattstellung, Round-Trip-Tracking mit
    Bar-Timestamps) — die Lauf-1/2-Zahlen sind mit den Lauf-3-Zahlen
    **nicht vergleichbar**.

    **Dritter Kalibrierungslauf (03.09.2026, korrigierte Engine + Exit-
    Regeln):** Auf Engine-Ebene wurden zwei Exit-Regeln ergänzt (wirkt auf
    alle Strategien, auch im Gate-Sweep): ``--stop-loss`` (Exit bei
    ``close <= avg_price × (1 − stop)``) und ``--max-holding-bars``
    (Exit nach N Kerzen). Die Glattstellung läuft jetzt zum Marktpreis,
    Positionen werden mark-to-market bewertet, und die Metriken zählen
    echte Round-Trips (``trade_data`` aus Close-Fills, Timestamps aus den
    Bars). Der Report zeigt pro Szenario zusätzlich die Exit-Verteilung
    nach Exit-Grund.

    Stop-Loss-Sweep auf crash-2021-05 (5m, Gate 0,3, Max-Haltezeit
    2016 Bars ≈ 7 Tage, Startkapital 100.000 $, Long-Only):

    | Stop-Loss | Return | Win-Rate | Profit-Factor | Round-Trips |
    |---:|---:|---:|---:|---:|
    | 4 % | -0,8 % | 38,0 % | 0,97 | 50 |
    | 6 % | -1,3 % | 38,0 % | 0,80 | 50 |
    | 8 % | -1,3 % | 38,0 % | 0,81 | 50 |
    | 12 % | -1,0 % | 39,6 % | 0,88 | 48 |

    Kein Stop-Wert dreht das Szenario ins Plus; der Stop ist bei 5m-
    Auflösung kaum relevant (die SHORT-Konsens-Exits machen 45 von 50
    Round-Trips und sind für sich leicht positiv, +728 $ bei 4 % Stop).
    Für den finalen Lauf wurde 8 % als robuster Mittelwert gewählt.

    Finaler E2E-Lauf (1m, alle vier Szenarien, Gate 0,3, Stop 8 %,
    Max-Haltezeit 10080 Bars ≈ 7 Tage):

    | Szenario | Final Equity | Return | Win-Rate | Round-Trips | Profit-Factor | Max-DD |
    |---|---:|---:|---:|---:|---:|---:|
    | crash-2021-05 (11 Tage) | 89.520 $ | -10,5 % | 19,9 % | 236 | 0,26 | 10,6 % |
    | pump-2021-11 (9 Tage) | 94.724 $ | -5,3 % | 14,0 % | 236 | 0,39 | 5,3 % |
    | crash-2022-06 (10 Tage) | 94.245 $ | -5,8 % | 18,7 % | 246 | 0,72 | 5,8 % |
    | range-2022-03 (31 Tage) | 86.126 $ | -13,9 % | 16,0 % | 649 | 0,65 | 14,6 % |

    Befund: Die korrigierte Abrechnung bestätigt die Bucket-Schätzung des
    zweiten Laufs (Win-Rate 14–20 % statt des damaligen Artefakt-Werts
    „0 %", hundert bis sechshundert echte Round-Trips pro Szenario). Der
    8-%-Stop löst praktisch nie aus (einmal in crash-2021-05), die
    Max-Haltezeit nie — die SHORT-Konsens-Exits dominieren mit über 99 %
    aller Closes, und ihr mittleres PnL ist in allen vier Szenarien
    negativ (-5 bis -21 $ pro Trade). Die Exit-Regeln haben den Verlust
    also nicht beseitigt; der Engpass bleibt die **Entry-Seite**: Das
    LONG-Konsens-Feuern kauft systematisch nahe lokaler Hochs, und bei
    1m-Auflösung verzehnfacht die Trade-Frequenz (236–649 Round-Trips)
    den Kostenabrieb. Der Resampling-Effekt dominiert: dasselbe
    crash-2021-05-Szenario liefert bei 5m nur -0,8 bis -1,3 % (nahe
    Break-even), bei 1m -10,5 %. Der Agenten-Konsens hat in diesen
    BTC/USDT-Szenarien (2021/2022) keinen positiven Long-Only-Edge;
    nächster sinnvoller Hebel ist die Entry-Selektion (z. B. nur
    kaufen, wenn Trend- und Volatilitäts-Regime übereinstimmen), nicht
    weitere Exit-Tuning.

    **Vierter Kalibrierungslauf (03.09.2026, Entry-Selektion):** Aus dem
    Befund des dritten Laufs (Engpass = Entry-Seite, LONG-Feuern kauft nahe
    lokaler Hochs) wurden zwei Entry-Filter auf der Strategie-Ebene ergänzt
    (wirkt nur auf die BUY-Seite, Exits bleiben unverändert):
    ``--entry-gate`` (zusätzliches BUY-Gate, das das Basis-Gate nur erhöht:
    bei 0,6 müssen drei von vier Agenten auf der LONG-Seite übereinstimmen)
    und ``--entry-required-agents`` (Fail-Closed-Pflichtvoten, z. B.
    ``trend,volatility_regime`` — alle genannten Agenten müssen in derselben
    Evaluation selbst LONG votieren; fehlt ein Agent oder votiert er nicht
    LONG, wird der Kauf blockiert). Beide Flags erben auch der Gate-Sweep
    und die Replay-Rechnung.

    Entry-Varianten-Sweep auf crash-2021-05 (5m, Gate 0,3, Stop 8 %,
    Max-Haltezeit 2016 Bars, Startkapital 100.000 $, Long-Only):

    | Variante | Return | Win-Rate | Profit-Factor | Round-Trips | Max-DD |
    |---|---:|---:|---:|---:|---:|
    | Baseline (ohne Entry-Selektion) | -1,3 % | 38,0 % | 0,81 | 50 | 2,40 % |
    | Entry-Gate 0,6 | +0,02 % | 45,5 % | 1,25 | 22 | 0,71 % |
    | Pflicht: trend | -1,6 % | 17,9 % | 0,46 | 28 | 2,30 % |
    | Pflicht: trend, volatility_regime | -0,01 % | 25,0 % | 1,11 | 4 | 0,20 % |
    | Entry-Gate 0,6 + Pflicht: trend | -0,1 % | 0,0 % | 0,00 | 2 | 0,11 % |
    | Pflicht: mean_reversion | +0,00 % | 59,1 % | 1,29 | 22 | 0,45 % |

    Die Trend-Pflicht verschlechtert das Szenario sogar (die Strategie
    kauft ohne Trendbestätigung seltener und besser als mit ihr) — das
    bestätigt „LONG kauft nahe lokaler Hochs". Das Entry-Gate 0,6 wird als
    robuster Gewinner gewählt (nahe Break-even mit 22 Round-Trips,
    agenten-agnostisch).

    Finaler E2E-Lauf (Entry-Gate 0,6, Gate 0,3, Stop 8 %; 5m mit
    Max-Haltezeit 2016 Bars, 1m mit 10080 Bars):

    5m (4× weniger Kostenabrieb, gröberes Fenster):

    | Szenario | Return | Win-Rate | Round-Trips | Profit-Factor | Max-DD |
    |---|---:|---:|---:|---:|---:|
    | crash-2021-05 (11 Tage) | +0,02 % | 45,5 % | 22 | 1,25 | 0,71 % |
    | pump-2021-11 (9 Tage) | -0,57 % | 7,1 % | 28 | 0,29 | 0,62 % |
    | crash-2022-06 (10 Tage) | -0,51 % | 30,0 % | 30 | 0,66 | 0,71 % |
    | range-2022-03 (31 Tage) | -0,35 % | 29,6 % | 81 | 1,31 | 0,48 % |

    1m (vergleichbar mit dem finalen Lauf 3):

    | Szenario | Final Equity | Return | Win-Rate | Round-Trips | Profit-Factor | Max-DD | (Lauf 3 ohne Entry-Selektion) |
    |---|---:|---:|---:|---:|---:|---:|---:|
    | crash-2021-05 (11 Tage) | 97.073 $ | -2,9 % | 21,8 % | 133 | 0,37 | 3,1 % | -10,5 % |
    | pump-2021-11 (9 Tage) | 98.508 $ | -1,5 % | 17,7 % | 136 | 0,26 | 1,5 % | -5,3 % |
    | crash-2022-06 (10 Tage) | 98.250 $ | -1,8 % | 23,0 % | 161 | 0,76 | 1,8 % | -5,8 % |
    | range-2022-03 (31 Tage) | 96.570 $ | -3,4 % | 21,6 % | 371 | 0,84 | 3,8 % | -13,9 % |

    Befund: Das Entry-Gate 0,6 ist der größte Hebel aller vier Läufe: Es
    halbiert die Round-Trips (133–371 statt 236–649 bei 1m) und senkt die
    Verluste von -5 bis -14 % auf -1,5 bis -3,4 % (crash-2021-05: von
    -10,5 % auf -2,9 %); die Max-Drawdowns fallen von 5–15 % auf 1,5–3,8 %.
    Bei 5m-Auflösung liegen alle vier Szenarien nahe Break-even (-0,6 % bis
    +0,02 %) mit Drawdowns unter 0,8 %. Die Strategie ist damit von
    „klar negativ mit hohen Drawdowns" zu „niedriges Risiko, nahe
    Break-even" verbessert — aber es entsteht **kein klar positiver
    Edge**: Die verbleibenden Verluste sind der Kostenabrieb
    (Slippage/Commission 0,1 % pro Seite), den die schwache verbleibende
    PnL nicht mehr vollständig deckt. Die Trend-Pflicht-Voten sind als
    Entry-Filter ungeeignet (sie bestätigen Hochkäufe); die
    Mean-Reversion-Pflicht ist nahe Break-even mit hoher Win-Rate, aber zu
    restriktiv in Kombination (1–4 Round-Trips). Sinnvolle nächste Hebel:
    den Handel auf 5m-Auflösung mit Entry-Gate verlagern (geringerer
    Kostenabrieb), die Trade-Frequenz weiter drosseln oder eine echte
    Alpha-Quelle ergänzen — weiteres Gate- oder Exit-Tuning hat nach vier
    Läufen keinen positiven Effekt mehr gezeigt.

    **Fünfter Kalibrierungslauf (04.09.2026, Strategie-Zoo: Bibliothek +
    LLM-Prompt):** Statt dem Agenten-Konsens wurden zwei unabhängige
    Strategiewege gegenübergestellt (jeweils derselbe Engine-Pfad,
    identische Kosten/Abrechnung): (a) eine deterministische
    Strategie-Bibliothek (`packages/strategies/`, 10 öffentlich
    dokumentierte Regel-Strategien: EMA-/MACD-Cross, Supertrend,
    Donchian-/Keltner-Breakout, RSI-/Bollinger-/VWAP-Mean-Reversion,
    Stochastik, ROC-Momentum — Long-Only, 10-%-Flatsize, 300-Bar-
    Fenster) und (b) die `PromptStrategy` (LLM `local-fast` via
    `litellm` trifft die Handelsentscheidungen selbst).

    Bibliotheks-Sweep (5m, alle vier Szenarien, 40 Runs, je Strategie ×
    Szenario; Rückgabe in %):

    | Strategie | crash-2021-05 | pump-2021-11 | crash-2022-06 | range-2022-03 | Σ |
    |---|---:|---:|---:|---:|---:|
    | rsi_mean_reversion | **+1,08** | -0,16 | **+0,93** | -0,36 | **+1,49** |
    | vwap_reversion | +0,79 | -0,38 | -1,62 | -0,77 | -1,98 |
    | ema_cross | -0,26 | -0,20 | -0,12 | -1,05 | -1,63 |
    | supertrend | -0,34 | -0,16 | -0,24 | -0,55 | -1,29 |
    | keltner_breakout | -1,19 | +0,33 | -0,77 | -0,34 | -1,97 |
    | bollinger_reversion | -1,22 | -0,37 | -0,48 | -1,33 | -3,40 |
    | macd_cross | -1,29 | -0,63 | -0,66 | -1,84 | -4,42 |
    | stochastics | -1,46 | -1,12 | -1,85 | -2,64 | -7,07 |
    | momentum_roc | -5,14 | +0,42 | -1,75 | -1,74 | -8,21 |
    | donchian_breakout | -5,43 | +0,03 | -1,14 | -1,90 | -8,44 |

    `rsi_mean_reversion` ist der einzige Σ-positive Kandidat
    (Win-Rate 78–83 %, 9–30 Round-Trips) — und der einzige, der in
    **beiden** Crash-Szenarien deutlich positiv ist. Kalibrierung
    (2021) vs. Validierung (2022) bleiben dabei konsistent: +0,92 %
    (2021) vs. +0,57 % (2022) im Σ der jeweiligen Szenarien.
    Regime-Muster: Mean-Reversion gewinnt in den Crashes, Trendfolge
    (Momentum/Keltner/Donchian) gewinnt im Pump — range-2022-03 ist für
    alle 10 Strategien negativ (kostengetriebener Seitwärtsabrieb, bei
    5m-Auflösung -0,3 % bis -2,6 %). Zum Vergleich: der 4. Lauf
    (Agenten-Ensemble mit Entry-Gate 0,6, 5m) lag bei -1,41 % Σ. Der
    beste Zoo-Kandidat (rsi_mean_reversion, +1,49 %) schlägt damit die
    Ensemble-Baseline deutlich, 9 von 10 Strategien (Mittel -3,69 %)
    nicht — es bleibt **kein robust positiver Edge** (nur 1 von 10
    Strategien Σ-positiv, im Validierungsjahr 2022 nur noch +0,57 %).

    PromptStrategy-E2E (crash-2021-05, 5m, `local-fast`, `--llm-every
    60`, Warmup 120): 51 LLM-Aufrufe, 0 Fehler, 1 BUY + 3 SELL-Signale
    → 1 Round-Trip, -0,19 % (Max-DD 0,53 %). Die Pipeline ist voll
    funktionsfähig (Snapshot → LLM → Signal → Engine → Artefakte +
    JSONL-Cache für deterministische Re-Runs); die Handelsqualität liegt
    auf dem Niveau der übrigen Strategien (kein Edge). Für 1m-Auflösung
    ist der Prompt-Modus im aktuellen Setup unpraktikabel: ein 1m-
    Replay desselben Szenarios hätte 5× so viele Bars (15.840) und bei
    gleichem Rhythmus (~5 Stunden/Aufruf) 5× so viele sequentielle
    LLM-Aufrufe (~260); der Engine-Teil wächst wegen der O(n²)-
    Indikatoren zusätzlich überproportional. 5m ist daher die
    sinnvolle Auflösung für den Prompt-Modus.

    Nebenbefund (Performance): `compute_indicators` in
    `packages/backtesting/datafeed.py` ist O(n²) (EMA/RSI/MACD werden
    pro Bar vom ersten Index neu iteriert) — ein 31-Tage-Szenario
    (8.928 5m-Kerzen) kostet dadurch ~3,5 min pro Backtest statt ~25 s
    bei 11 Tagen. Kandidat für eine Folge-Optimierung (einmaliger
    rekursiver Durchlauf, Werte unverändert).

    **Sechster Kalibrierungslauf (04.09.2026, OOS-Validierung +
    Engine-Korrektur: Flatsize):** Vor der Validierung auf weiteren
    Zeiträumen wurden zwei Engine-Bugs gefunden und korrigiert:
    (1) **Pyramiding**: Die Regel-Strategien feuern BUY-Signale wiederholt
    (RSI kreuzt die Schwelle mehrfach), und die Engine addierte jede
    Bestellung zur bestehenden Position — dadurch wuchs eine 10-%-
    „Flatsize"-Strategie über Zeit bis zu ~100 % der Equity. Die Zoo-
    Zahlen des 5. Laufs sind damit nur mit Vorbehalt interpretierbar.
    Fix: `BacktestConfig.allow_pyramiding` (Default `True` = Verhalten
    der Läufe 1–5, Backward-Compat) + CLI-Flag `--no-pyramiding` für
    echten Flatsize (BUY bei offener Position wird ignoriert).
    (2) **Trade-Return-Metrik**: `avg/best/worst_*_trade_return_pct`
    rechneten Dollar-PnL × 100 statt dem Anteil am Handelsnotional;
    `entry_price` wird jetzt in `trade_data` mitgeführt. Zusätzlich
    wurde der O(n²)-Nebenbefund umgesetzt: `compute_indicators` rechnet
    EMA/MACD jetzt in einem rekursiven Durchlauf (bit-identisch zu
    vorher, E2E-Äquivalenz replays den 4. Lauf exakt; 158.000 Bars:
    >10 min → 2,0 s).

    Datenbasis: `candles_history` (BTC/USDT, BINANCE_FUTURES) deckt
    zusätzlich den Zeitraum **2026-03-02 → 2026-09-02** ab (185 Tage,
    echte Out-of-Sample-Daten, nie für Kalibrierung verwendet; Lücken
    in der Historie: 2021-11-30 → 2022-02-15, 2022-07-31 → 2026-03-02).

    RSI-Parameter-Sweep (5m, 125 Configs × 4 Szenarien) identifizierte
    `period=30` als robusten Kandidaten; die Flatsize-Validierung
    darüber auf **10 definierten Fenstern** (10 % Position,
    `allow_pyramiding=False`, Kosten 0,1 %/Seite, 100.000 $):

    | Fenster | Zeitraum | Regime |
    |---|---|---|
    | crash-2021-05 / pump-2021-11 | 05-15→05-25 / 11-01→11-10, 2021 | Kalibrierung |
    | range-2022-03 / crash-2022-06 | 03-01→03-31 / 06-15→06-25, 2022 | Validierung |
    | drop-2021-06 | 05-26→06-30, 2021 | OOS |
    | bear-2022-q1 / range-2022-q2 / luna-2022 | 02-15→04-15 / 04-15→06-14 / 06-15→07-31, 2022 | OOS |
    | oos-2026-h1 / oos-2026-h2 | 03-02→06-15 / 06-15→09-02, 2026 | OOS (nie kalibriert) |

    | Kandidat | Σ | positiv | worst | Max-DD |
    |---|---:|---:|---:|---:|
    | A: rsi p30/b30/s80 (Zoo-Default) | **-6,46** | 5/10 | -4,85 | 6,23 |
    | D: rsi p30/b20/s80 | **+4,06** | **6/10** | -1,80 | 4,09 |
    | E: rsi_vol_gate p30/b30/s80 | +3,91 | 5/10 | -1,82 | 3,07 |

    Befund: (1) Der Zoo-„Gewinner" des 5. Laufs (A, +1,49 %) kollabiert
    unter echtem Flatsize auf Σ -6,46 % — seine damalige Robustheit war
    pyramiding-getrieben. (2) **D** (nur tiefe Oversold: RSI30 < 20)
    ist der robusteste Kandidat: 6/10 Fenster positiv, in den
    2026-OOS-Hälften +1,20 % bzw. +0,97 %, worst Window -1,80 %
    (range-2022-q2, LUNA-Crash; Buy-&-Hold dort -44,6 %). (3) **E**
    (neue Strategie `rsi_vol_gate`: RSI-Reversion + ATR-Gate, BUY nur
    bei ATR/close ≥ 0,8 %) hat das niedrigste Max-DD, handelte aber in
    beiden 2026-Fenstern gar nicht (Gate zu restriktiv im ruhigen
    Regime). (4) Ehrliche Einordnung: D erzeugt über ~446 Fenster-
    tage nur 13 Round-Trips — statistisch dünn; der Mechanismus
    (kaufen nur bei extremem Oversold-Ausschlag) ist aber konsistent
    positiv, schlägt Buy-&-Hold in allen Crash-/Bear-Fenstern um
    Größenordnungen und unterperformt strukturell nur in starken
    Bullen (oos-2026-h2: +0,97 % vs. B&H +18,3 %). **Verdikt: kein
    deployment-reifer Edge, aber D ist der erste Kandidat mit
    OOS-bestätigtem, mechanismusplausibel positivem Flatsize-Ergebnis.**
    Nächster Hebel: Regime-Router (Mean-Reversion in Crash/Range,
    Trendfolge im Pump — das im 5. Lauf beobachtete Regime-Muster).

Beide Services liegen hinter dem Compose-Profil `on-demand` — sie starten
nie mit `docker compose up`, nur explizit via `docker compose run`.

### Hinweise

- **Agenten im Realbetrieb**: Der Orchestrator läuft standardmäßig mit
  `ACTIVE`-Agenten (`ORCHESTRATOR_AGENT_STATUS=ACTIVE`; auf `SHADOW`
  setzbar für den reinen Beobachtungsmodus). Der Demo-Modus führt
  Konsens-Entscheidungen als **virtuelle** Paper-Orders aus
  (Konfidenz-Gate ≥ 0,3, long-only). `live_trading_enabled` bleibt
  deaktiviert — es werden **nie** echte Orders ausgeführt.
- **Demo-Modus (imaginäres Geld)**: `demo-trader` führt echte
  Konsens-Entscheidungen der Agenten auf Binance-Futures-Kursen als
  Paper-Trades aus (long-only, max. 10 % Position, Slippage/Commission
  0,1 %). Alles sichtbar im Web-Dashboard unter
  `http://localhost:8080/` (Konto, Positionen, Trades, Entscheidungen,
  News).
- **Zentrale Web-UI**: `http://localhost:8080/` bündelt alle
  Oberflächen als Tabs — Trading (eigenes Dashboard), Monitoring
  (Grafana), Metriken (Prometheus), Alerts (Alertmanager), ML (MLflow)
  und Storage (Minio). Die fremden UIs laufen über einen Reverse-Proxy
  der API (`/proxy/...`), damit sie embedded funktionieren; die
  Originale bleiben unter ihren bisherigen Ports erreichbar.
  Alle eingebetteten Oberflächen sind **login-frei** (lokaler Dev-Stack,
  nur localhost): Grafana läuft mit anonymem Admin-Zugriff, der
  Storage-Tab meldet sich bei MinIO automatisch an (Credentials bleiben
  im API-Container). Das Monitoring-Tab zeigt das provisionierte
  Dashboard „Trading Orchestra — System" (API-, Redpanda- und
  ClickHouse-Panels); der ML-Tab wird pro Demo-Zyklus mit einem
  MLflow-Run gefüllt (Entscheidung, Konfidenz, Konto). Live-Updates
  über WebSocket sind im eingebetteten Modus deaktiviert (der Proxy
  forwardet keine WS-Upgrade-Requests) — die Panels aktualisieren sich
  per Polling/Refresh.
- Die mitgelieferten RSS-Feeds (CoinDesk, Cointelegraph, Decrypt, The Block,
  Bitcoin Magazine, Crypto Potato) sind auf Erreichbarkeit geprüft; neue
  Events landen pro Zyklus in `news_events` (sichtbar unter
  `/status` → `streaming.news_events_total`).
- **Autostart nach Reboot**: `ops/systemd/install_autostart.sh` (sudo)
  installiert eine systemd-Unit, die den Stack beim Boot startet.
  `restart: unless-stopped` deckt nur Docker-Daemon-Neustarts ab.
- Secrets liegen in `configs/secrets/*.txt` (gitignored) und werden über
  Docker Secrets in die Container eingelesen.
- Test-Gate: `.venv/bin/pytest tests/unit -q` (Lint-Gate: `uvx ruff check`
  über alle tracked Files).
