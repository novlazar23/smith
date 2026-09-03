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
