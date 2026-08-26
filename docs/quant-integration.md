# Quant-Plattform Integration Guide

## Shadow Trading Loop Integration

Die Quant-Plattform integriert sich in den Shadow Trading Loop über den
`EvidenceAggregator`, der alle Quant-Evidence zu einem einheitlichen Dict
für den Trading-Orchestrator zusammenführt.

### Integrationsschritte

1. **OHLCV-Ingestion**: Marktdaten werden über `/quant/ingest/ohlcv` in InfluxDB geschrieben

2. **Feature-Extraktion**: Für jedes Symbol werden Features berechnet
   ```python
   from trading_harness.quant.features import FeatureEngine
   engine = FeatureEngine()
   features = engine.compute(candles)
   ```

3. **Anomalie-Erkennung**: Anomalien werden pro Tick erkannt
   ```python
   from trading_harness.quant.anomaly_detection import AnomalyDetectionEngine
   engine = AnomalyDetectionEngine()
   anomalies = engine.detect(candles)
   ```

4. **Regime-Erkennung**: Das aktuelle Marktregime wird bestimmt
   ```python
   from trading_harness.quant.regime_detection import RegimeDetectionEngine
   engine = RegimeDetectionEngine()
   regime = engine.detect(candles)
   ```

5. **Evidence Aggregation**: Alle Evidence wird zusammengeführt
   ```python
   from trading_harness.quant.evidence_aggregator import EvidenceAggregator
   aggregator = EvidenceAggregator()
   aggregator.add_entry("features", features)
   aggregator.add_entry("anomalies", anomalies)
   aggregator.add_entry("regime", regime)
   evidence = aggregator.aggregate(symbol, timeframe)
   ```

6. **Trading-Entscheidung**: Der Orchestrator nutzt die Evidence

### API-Integration

```python
import httpx

# Features berechnen
response = httpx.post("http://localhost:8080/quant/features/compute", json={
    "symbol": "BTCUSDT",
    "timeframe": "1m",
    "exchange": "binance",
    "candles": candles,
})
features = response.json()["features"]

# Anomalien erkennen
response = httpx.post("http://localhost:8080/quant/anomalies/detect", json={
    "symbol": "BTCUSDT",
    "timeframe": "1m",
    "exchange": "binance",
    "candles": candles,
})
anomalies = response.json()["anomalies"]

# Regime bestimmen
response = httpx.post("http://localhost:8080/quant/regime/detect", json={
    "symbol": "BTCUSDT",
    "timeframe": "1m",
    "exchange": "binance",
    "candles": candles,
})
regime = response.json()["regime"]
```

### Shadow-Loop-Status prüfen

```bash
curl http://localhost:8080/quant/shadow/status
```

Response:
```json
{
  "status": "ok",
  "integration_active": true,
  "quant_engines": ["features", "anomalies", "regime", "similarity", "forward_outcomes", "ml_features", "backtest"],
  "last_evidence": {...}
}
```

## Performance-Optimierung

### Feature Cache

```python
from trading_harness.quant.feature_cache import FeatureCache

cache = FeatureCache(max_size=1000, default_ttl=300.0)

# Features mit Cache
features = cache.get("BTCUSDT_1m_features")
if features is None:
    features = engine.compute(candles)
    cache.put("BTCUSDT_1m_features", features)

# Cache-Statistiken
stats = cache.stats()
print(f"Hit Rate: {stats.hit_rate:.2%}")
```

### Batch-Verarbeitung

```python
from trading_harness.quant.batch_processor import BatchProcessor

processor = BatchProcessor(chunk_size=10)
job_id = processor.create_job(["BTCUSDT", "ETHUSDT", "SOLUSDT"])

def process_batch(symbols):
    return {s: engine.compute(get_candles(s)) for s in symbols}

result = processor.process_chunked(job_id, process_batch)
print(f"Processed {result.processed}/{result.total}")
```

## Fehlerbehandlung

### Input-Validierung

```python
from trading_harness.quant.validation import Validator

validator = Validator()
result = validator.validate_candle(candle)
if not result.valid:
    print(f"Invalid candle: {result.errors}")
```

### Error Recovery

```python
from trading_harness.quant.error_recovery import ErrorRecovery, RetryConfig

recovery = ErrorRecovery(RetryConfig(max_retries=3, base_delay=1.0))
result = recovery.with_retry(
    lambda: compute_features(candles),
    fallback={},
    operation_name="feature_computation",
)
if result.success:
    features = result.value
else:
    print(f"Failed after {result.retries} retries: {result.error}")
```
