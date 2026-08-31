# Trading Orchestra

## Running the system

Vollständiger Stack mit einem Befehl (bootstrapped automatisch):

```bash
docker compose up -d --build
```

`db-init` (One-Shot) legt bei jedem Start idempotent an: PostgreSQL-Tabellen
(`trading`), ClickHouse-Tabellen (`trading_events`), Redpanda-Topic
`market_data`. Die App-Services starten erst nach dessen Abschluss.

### Laufende Services

| Service | Aufgabe | Persistiert in |
|---|---|---|
| `api` | REST-API auf `localhost:8080` (`/status`, `/metrics`, `/v1/...`) | — |
| `market-producer` | Synthetische Candles (Dummy-Adapter, 60 s Ticks) | Redpanda `market_data` |
| `ingestion-consumer` | Konsument von `market_data`, validiert Candles | ClickHouse `trading_events.candles` |
| `news-ingestion` | RSS-Zyklen (30 s), dedupliziert und klassifiziert | PostgreSQL `news_events` |
| `alertmanager` | Alert-Ziel von Prometheus (`127.0.0.1:9093`) | — |

Dazu: `postgres`, `clickhouse`, `redis`, `minio`, `redpanda`, `mlflow`,
`prometheus` (`127.0.0.1:9090`), `grafana` (`127.0.0.1:3000`).

### Verifikation

```bash
curl -s localhost:8080/status | python3 -m json.tool      # "status": "running"
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

- **SHADOW-Phase**: `live_trading_enabled` ist deaktiviert; es werden keine
  echten Orders ausgeführt.
- Die 7 mitgelieferten RSS-Quellen sind aus dieser Umgebung derzeit nicht
  erreichbar (Redirects/DNS/403); der Ingestion-Loop läuft trotzdem und
  persistiert automatisch, sobald eine Quelle antwortet.
- Secrets liegen in `configs/secrets/*.txt` (gitignored) und werden über
  Docker Secrets in die Container eingelesen.
- Test-Gate: `.venv/bin/pytest tests/unit -q`.

## Running the system

Der komplette Stack startet mit einem Befehl — Schemas, Redpanda-Topic und alle
Services werden automatisch gebootstrapped:

```bash
docker compose up -d --build
```

`db-init` (One-Shot) legt bei jedem Start idempotent an:
PostgreSQL-Tabellen (`trading`), ClickHouse-Tabellen (`trading_events`),
Redpanda-Topic `market_data`. Die App-Services starten erst danach.

### Laufende Services

| Service | Aufgabe | Persistiert in |
|---|---|---|
| `api` | REST-API auf `localhost:8080` (`/status`, `/metrics`, `/v1/...`) | — |
| `market-producer` | Synthetische Candles (Dummy-Adapter, BTC/ETH, 60 s Ticks) | Redpanda `market_data` |
| `ingestion-consumer` | Konsument von `market_data`, validiert Candles | ClickHouse `trading_events.candles` |
| `news-ingestion` | RSS-Zyklen (30 s), dedupliziert + klassifiziert | PostgreSQL `news_events` |
| `alertmanager` | Alert-Ziel von Prometheus auf `127.0.0.1:9093` | — |

Dazu: `postgres`, `clickhouse`, `redis`, `minio`, `redpanda`, `mlflow`,
`prometheus` (`127.0.0.1:9090`), `grafana` (`127.0.0.1:3000`).

### Verifikation

```bash
curl -s localhost:8080/status | python3 -m json.tool   # "status": "running"
curl -s localhost:8080/metrics | grep -m1 http_requests_total
docker compose exec -T postgres psql -U orchestra -d trading -tc "SELECT count(*) FROM news_events"
docker compose exec -T clickhouse wget -qO- --header="X-ClickHouse-Database: trading_events" \
  --post-data "SELECT count() FROM candles" http://127.0.0.1:8123/   # wächst pro Producer-Tick
curl -s localhost:9093/api/v2/status | grep -o '"status":"ready"'
curl -s localhost:9090/api/v1/rules | grep -o 'trading-orchestra-alerts'
```

Hinweise:
- **SHADOW-Phase**: `live_trading_enabled` ist deaktiviert; es werden keine
  echten Orders ausgeführt.
- Die 7 mitgelieferten RSS-Feeds sind zurzeit von dieser Umgebung aus nicht
  erreichbar (Redirects/DNS/403); der Ingestion-Loop läuft trotzdem und
  persistiert automatisch, sobald die Feeds antworten.
- Test-Gate: `tests/unit` (venv: `.venv/bin/pytest tests/unit -q`).
