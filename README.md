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
