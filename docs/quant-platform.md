# Quantitative Trading Data & Analysis Platform

## Architektur-Übersicht

Die Quant-Plattform ist eine modulare Erweiterung des Trading Harness für
marktanalytische Berechnungen. Sie speichert marktdaten in InfluxDB,
extrahiert Features, erkennt Anomalien und Regime, findet historische
Ähnlichkeiten, berechnet Forward Outcomes und stellt ML-Features bereit.

### Kernmodule

| Modul | Beschreibung | Tests |
|-------|-------------|-------|
| `schema.py` | InfluxDB-Schema-Definition | 6 |
| `influxdb_client.py` | InfluxDB-Store (read/write/query) | 12 |
| `ohlcv_ingestion.py` | OHLCV-Daten-Import | 8 |
| `features.py` | Feature-Extraktion (RSI, MACD, Bollinger, ATR, VWAP) | 16 |
| `feature_store.py` | Feature-Persistenz | 10 |
| `anomaly_detection.py` | Anomalie-Erkennung (Z-Score, IQR, Price Shock) | 12 |
| `anomaly_store.py` | Anomalie-Persistenz | 8 |
| `regime_detection.py` | Regime-Erkennung (SMA, ADX, Volatility) | 12 |
| `regime_store.py` | Regime-Persistenz | 8 |
| `similarity.py` | Similarity Engine (Euclidean, Pearson) | 17 |
| `similarity_store.py` | Similarity-Persistenz | 9 |
| `forward_outcomes.py` | Forward Outcome Statistics | 11 |
| `forward_outcomes_store.py` | Forward Outcomes Persistenz | 8 |
| `ml_features.py` | ML Feature Builder | 12 |
| `feature_importance.py` | Feature Importance Engine | 11 |
| `backtesting.py` | Backtesting Engine | 13 |
| `backtest_store.py` | Backtest Persistenz | 18 |
| `evidence_aggregator.py` | Evidence Aggregation | 12 |
| `feature_cache.py` | LRU-Cache mit TTL | 11 |
| `batch_processor.py` | Batch-Verarbeitung | 10 |
| `validation.py` | Input-Validierung | 14 |
| `error_recovery.py` | Error-Recovery & Retry | 9 |
| `observability.py` | Metriken & Logging | 12 |

### API-Endpunkte

| Endpoint | Methode | Beschreibung |
|----------|---------|-------------|
| `/quant/ingest/ohlcv` | POST | OHLCV-Daten importieren |
| `/quant/schema` | GET | Schema-Informationen |
| `/quant/status` | GET | System-Status |
| `/quant/features/compute` | POST | Features berechnen |
| `/quant/features/{symbol}` | GET | Features abrufen |
| `/quant/anomalies/detect` | POST | Anomalien erkennen |
| `/quant/anomalies/{symbol}` | GET | Anomalien abrufen |
| `/quant/regime/detect` | POST | Regime erkennen |
| `/quant/regime/{symbol}` | GET | Regime abrufen |
| `/quant/similarity/find` | POST | Ähnliche Muster finden |
| `/quant/similarity/{symbol}` | GET | Similarity-Daten abrufen |
| `/quant/outcomes/compute` | POST | Forward Outcomes berechnen |
| `/quant/outcomes/{symbol}` | GET | Forward Outcomes abrufen |
| `/quant/ml/features` | POST | ML-Features erstellen |
| `/quant/ml/importance` | POST | Feature Importance berechnen |
| `/quant/backtest/run` | POST | Backtest ausführen |
| `/quant/backtest/{symbol}` | GET | Backtest-Ergebnisse abrufen |
| `/quant/shadow/status` | GET | Shadow-Loop-Status |
| `/quant/perf/cache-stats` | GET | Cache-Statistiken |
| `/quant/perf/batch-status` | GET | Batch-Job-Status |
| `/quant/validate` | POST | Daten validieren |

## Konfiguration

### InfluxDB

```env
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=my-super-secret-token
INFLUXDB_ORG=trading-harness
INFLUXDB_BUCKET=market-data
INFLUXDB_ENABLED=true
```

### Feature Cache

```python
from trading_harness.quant.feature_cache import FeatureCache

cache = FeatureCache(max_size=1000, default_ttl=300.0)
```

### Batch Processor

```python
from trading_harness.quant.batch_processor import BatchProcessor

processor = BatchProcessor(chunk_size=10)
job_id = processor.create_job(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
processor.process(job_id, lambda symbol: compute_features(symbol))
```

## Beispiele

### Features berechnen

```python
from trading_harness.quant.features import FeatureEngine

engine = FeatureEngine()
candles = [...]  # OHLCV-Daten
features = engine.compute(candles)
# {"rsi_14": 65.2, "macd": 1.23, "bb_upper": 105.0, ...}
```

### Anomalien erkennen

```python
from trading_harness.quant.anomaly_detection import AnomalyDetectionEngine

engine = AnomalyDetectionEngine()
anomalies = engine.detect(candles)
# [{"type": "price_shock", "severity": 0.8, "value": 0.15}, ...]
```

### Forward Outcomes berechnen

```python
from trading_harness.quant.forward_outcomes import ForwardOutcomeEngine

engine = ForwardOutcomeEngine(horizons=[5, 10, 20])
result = engine.compute(candles, pattern_length=10)
# ForwardOutcomeResult mit Hit Rate, Profit Factor, etc.
```

### Backtest ausführen

```python
from trading_harness.quant.backtesting import BacktestEngine

engine = BacktestEngine(initial_capital=10000)
strategy = engine.simple_moving_average_strategy(fast_period=10, slow_period=30)
result = engine.run(candles, strategy)
# BacktestResult mit PnL, Drawdown, Sharpe Ratio
```
