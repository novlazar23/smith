# Marktdatenfluss — Quant-Plattform Integration

> Phase 0 — Bestandsaufnahme  
> Erstellt: 2026-08-25

---

## 1. Aktueller Datenfluss (Smith Shadow Trading)

```
Exchange APIs (Bybit/Bitget/Binance/Coinbase)
    ↓ httpx (synchrone Adapter)
CryptoExchangeAdapter.get_ticker(symbol) → {"last": float, ...}
    ↓
CryptoExecutionRouter.get_ticker(symbol) — routing to correct adapter
    ↓
CryptoMarketDataProvider(router) —Wrapper, implementiert MarketDataProvider Protocol
    ↓
ShadowTradingLoop._market_data.get_ticker(symbol)
    ↓ asyncio.to_thread (synchron → async)
run_once():
    1. Ticker abrufen (pro Symbol)
    2. MarketSnapshot erzeugen
    3. Agent-Analyse (LLM-gesteuert)
    4. Aggregation
    5. Risk Engine
    6. Paper-Ausführung (PaperExecutionStack)
    7. ShadowTradingRecord
    8. Mark-to-Market
    9. Audit-Trail
```

### Kern-Protokoll

```python
# src/trading_harness/services/shadow_trading_loop.py, Zeile 93
class MarketDataProvider(Protocol):
    """Synchroner Ticker-Provider: get_ticker(symbol) -> {"last": float, ...}"""
    def get_ticker(self, symbol: str) -> dict[str, float]: ...
```

### Implementierung

```python
# crypto_exchange_adapter.py — 4 Adapter (Bybit/Bitget/Binance/Coinbase)
# crypto_execution_router.py — Single-Router mit lazy Adapter-Instanziierung
# shadow_trading_loop.py:110 — CryptoMarketDataProvider(router)
```

---

## 2. Integrationspunkte für Quant-Engine

### Option A: vor MarketDataProvider (rohe Ticker)
```
Exchange → Adapter → [QUANT: Rohdaten speichern] → Router → MarketDataProvider → Loop
```
- Vorteil: Volle Rohdaten (Trades, Orderbook) verfügbar
- Nachteil: Zusätzliche Latenz pro Tick

### Option B: als MarketDataProvider-Wrapper (empfohlen)
```
Exchange → Adapter → Router → [QUANT: Ticker anreichern] → MarketDataProvider → Loop
```
- Vorteil: Nur Ticker-Daten (nicht Trades/Orderbook), minimaler Overhead
- Nachteil: Quant-Engine sieht nur `last`-Preis, nicht Volume/Side

### Option C: nach run_once (Snapshot-Enrichment) ← Strategie B
```
Loop: run_once() → [QUANT: Snapshot anreichern] → Rest der Kette
```
- Vorteil: Quant-Engine arbeitet mit vollem Snapshot, keine Latenz pro Tick
- Nachteil: Nur periodic (pro Iteration, nicht pro Tick)

### Empfehlung: Option C (Snapshot-Enrichment)

**Begründung:**
1. Der Shadow-Loop läuft alle 900s (15min) — pro Iteration anreichern ist ausreichend
2. Quant-Engine braucht historischen Kontext, nicht Live-Tick-Latenz
3. Kein Performance-Impact auf Ticker-Abruf
4. Saubere Trennung: Quant-Engine liest aus InfluxDB, schreibt in Snapshot

---

## 3. Datenquellen für Quant-Plattform

### Direkt aus Smith (Phase 1)
| Daten | Quelle | Format |
|-------|--------|--------|
| OHLCV | `get_ticker()` × Zeitintervall | `{"last": float}` |
| Symbol | `settings.shadow_trading_symbols` | `list[str]` |
| Exchange | Crypto-Adapter-Konfiguration | `str` |
| Timestamp | `self._clock()` | `datetime` |

### Extern (Phase 2+, braucht eigene Ingestion)
| Daten | Quelle | Format |
|-------|--------|--------|
| Trades (Full) | Exchange WebSocket/API | `price, size, side` |
| Orderbook | Exchange API | `bids, asks` |
| Funding Rate | Exchange Derivatives API | `float` |
| Open Interest | Exchange Derivatives API | `float` |
| Liquidations | Exchange/Aggregator | `long, short` |
| Cross-Market | Yahoo Finance / TradingView | `OHLCV` |

---

## 4. Feature-Berechnungsfluss (Phase 2)

```
InfluxDB (OHLCV 1m)
    ↓ Batch-Query (letzte N Datenpunkte)
Feature-Engine
    ├── Trend: EMA, SMA, ADX, slope
    ├── Momentum: RSI, ROC, MACD
    ├── Volatilität: ATR, realized vol, percentile
    ├── Volume: z-score, imbalance
    └── Cross-Market: Korrelationen
    ↓
Market State Vektor X(t) ∈ R^n
    ↓ Normalisierung (Robust Scaling)
InfluxDB (features measurement)
    ↓
Historical Similarity Engine (Phase 5)
    ↓
Forward Outcome Statistics (Phase 6)
    ↓
Quant-Output API (Phase 9)
    ↓
Shadow Trading Loop (Snapshot-Enrichment)
```

---

## 5. Vollständigkeits-Checkliste

| Datenfluss | Status |
|------------|--------|
| Exchange → Adapter → Router | ✅ Implementiert |
| Router → MarketDataProvider | ✅ Implementiert |
| MarketDataProvider → Shadow-Loop | ✅ Implementiert |
| Shadow-Loop → Snapshot | ✅ Implementiert |
| **[NEU] InfluxDB → Feature-Engine** | ❌ Phase 2 |
| **[NEU] Feature-Engine → Market State** | ❌ Phase 2 |
| **[NEU] Market State → Historical Similarity** | ❌ Phase 5 |
| **[NEU] Forward Outcomes → Quant Output** | ❌ Phase 6 |
| **[NEU] Quant Output → Snapshot-Enrichment** | ❌ Phase 9 |
