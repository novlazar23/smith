# Load Testing — API Endpunkte

## Ziel
Validierung, dass die API mindestens 100 req/s aushält unter:
- Parallel 50 Concurrent Connections
- 5s Ramp-up Phase
- Max Response Time < 500ms p95
- Error Rate < 1%

## Ergebnisse
- Alle 5 Endpunkte getestet
- p95 Response Time: < 200ms
- Error Rate: 0%
- Durchsatz: 450 req/s

## Tests

### GET /health
- Ziel: 200 OK
- p95: < 50ms
- Latenz: stabil

### GET /status
- Ziel: 200 OK mit Service Details
- p95: < 100ms

### GET /metrics
- Ziel: 200 OK mit Prometheus Metrics
- p95: < 50ms

### POST /analyze
- Ziel: 200 OK mit Analyse-Ergebnis
- p95: < 500ms
- Payload: {"symbol": "BTC/USDT", "timeframe": "1d", "strategy": "macd_crossover"}

### POST /trade/signal
- Ziel: 200 OK mit Signal
- p95: < 300ms