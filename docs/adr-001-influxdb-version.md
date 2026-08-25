# ADR-001: InfluxDB Version

> Phase 0 — Bestandsaufnahme  
> Erstellt: 2026-08-25  
> Status: **Proposed**

---

## Kontext

Die Quant-Plattform benötigt einen Zeitreihenspeicher für OHLCV-Daten, Trades,
Orderbook-Snapshots und Derivate-Daten. InfluxDB ist die primäre Wahl als
zentraler Time-Series-Store.

Es muss entschieden werden: InfluxDB 2.x OSS oder InfluxDB 3.x (Apache Arrow /
DataFusion).

---

## Entscheidung

**Empfehlung: InfluxDB 2.7 OSS**

---

## Alternativen

### InfluxDB 2.7 OSS

| Aspekt | Bewertung |
|--------|-----------|
| Python Client | `influxdb-client` ≥1.40 — reif, gut dokumentiert,活跃 gewartet |
| Flux Query Language | Etabliert, ausreichend für OHLCV/Feature-Queries |
| Downsampling | Kapazity Tasks (kontinuierlich) oder Continuous Queries |
| Docker Image | `influxdb:2.7` — stabil, ~300 MiB |
| Speicher-Effizienz | TSME (Time-Structured Merge Tree) — gut für append-only |
| Community | Groß, viele Beispiele, Stack Overflow |
| LTS | 2.7 wird noch gewartet (Keystone-Release) |
| Upgrade-Pfad | 2.x → 3.x möglich wenn 3.x reif ist |

### InfluxDB 3.x (Apache Arrow / DataFusion)

| Aspekt | Bewertung |
|--------|-----------|
| Python Client | `influxdb-client-python` — neu, weniger stabil |
| Query Language | SQL + InfluxQL (neu) |
| Architektur | Separated Compute/Storage (Cloud-fokussiert) |
| OSS | `influxdb3` Community Edition — frühes Stadium |
| Docker Image | `influxdb3` — noch nicht production-ready |
| Vorteil | Kolonnenformat (Parquet), bessere Kompression |
| Risiko | Breaking Changes, unstable API, kleine Community |

### Andere Alternativen (verworfen)

| Option | Grund der Verwerfung |
|--------|---------------------|
| TimescaleDB | Smith nutzt bereits PostgreSQL — Overhead für reine Time-Series |
| QuestDB | Geringere Python-Client-Reife |
| Prometheus | Nicht für-finanzielle-Daten optimiert |
| ClickHouse | Overhead, zu komplex für MVP |
| SQLite + pandas | Kein nativer Time-Series-Support |

---

## Begründung

1. **Python-Client-Reife:** `influxdb-client` ist der stabilste Client für
   Zeitreihen-Datenbanksysteme in Python. async-Support vorhanden.

2. **Docker-Stabilität:** `influxdb:2.7` läuft zuverlässig, Health-Check
   funktioniert, Setup-Container (`influxdb-init`) für automatische Konfiguration.

3. **Downsampling:** Kapazity Tasks erlauben kontinuierliches Downsampling
   (1m → 5m/15m/1h/4h/1d) ohne externen Cron.

4. **Upgrade-Pfad:** Wenn InfluxDB 3.x production-ready ist, kann migriert
   werden. Die Query-Sprache (Flux) und das Datenmodell (Measurements/Tags/Fields)
   sind identisch.

5. **Risiko minimieren:** 2.7 ist eine etablierte Version mit vielen
   Produktions-Deployment. 3.x ist zu früh für ein MVP.

---

## Konsequenzen

### Positiv
- Schneller Start (Tag 1 produktiv)
- Guter Python-Client mit async-Support
- Etablierte Docker-Images
- Einfaches Downsampling via Tasks

### Negativ
- Flux ist eine proprietäre Query-Sprache (kein SQL)
- TSME ist less efficient als Arrow-Format für Analysen
- Mögliche Migration zu 3.x in 12-18 Monaten

### Risiken
- Flux-Lernkurve (aber gut dokumentiert)
- Performance bei sehr großen Datenmengen (>100M Punkte) — aber nicht relevant für MVP

---

## Reversal-Strategie

Wenn InfluxDB 3.x nach 12 Monaten production-ready ist:
1. Neues InfluxDB 3.x Docker-Image in docker-compose.yml
2. `influxdb-client` durch `influxdb-client-python` ersetzen
3. Flux-Queries durch SQL ersetzen (ähnliche Semantik)
4. Daten-Migration via Export/Import

**Aufwand:** ~2-3 Tage (Schema identisch, nur Query-Sprache ändert sich)

---

## Nächste Schritte

1. InfluxDB 2.7 zu docker-compose.yml hinzufügen (Phase 1, P1-1)
2. `influxdb-client` als optionale Dependency (Phase 1, P1-2)
3. InfluxDB-Health-Check implementieren (Phase 1, P1-4)
