# Arbeitsplan: Docker-Ready Autonomous System

> Quant Platform abgeschlossen (Phase 0-13, `3ced258`)
> Ziel: Autark in Docker lauffähiges System mit Web-Oberfläche

---

## Architektur

```text
+-----------------------------------------------------------+
|                    DOCKER CONTAINER                       |
|                                                           |
|  +------------------+  +------------------+              |
|  |   PostgreSQL     |  |   InfluxDB       |              |
|  |   (Datenbank)    |  |   (Zeitreihen)   |              |
|  +------------------+  +------------------+              |
|           |                     |                         |
|           v                     v                         |
|  +--------------------------------------------------+    |
|  |              TRADING HARNESS API                  |    |
|  |  - Agent Runtime    - Shadow Trading Loop         |    |
|  |  - Evolution        - Risk Engine                 |    |
|  |  - Quant Platform   - Execution Gateway          |    |
|  +--------------------------------------------------+    |
|           |                                               |
|           v                                               |
|  +--------------------------------------------------+    |
|  |              WEB INTERFACE (React)                |    |
|  |  - Dashboard         - Agent Management           |    |
|  |  - Shadow Trading    - Quant Analytics            |    |
|  |  - Performance       - System Status              |    |
|  +--------------------------------------------------+    |
|                                                           |
+-----------------------------------------------------------+
```

---

## Tasks

### T1: PostgreSQL Integration
**Was:** Echte PostgreSQL statt In-Memory Fallback
- Docker-Compose mit PostgreSQL Service
- SQLAlchemy Models für alle Tabellen
- Migrationen mit Alembic
- Tests mit Test-DB

### T2: Evaluation Engine
**Was:** Brier Score, Calibration, Walk-Forward
- `services/evaluation_engine.py`
- Metriken-Berechnung
- Out-of-Sample Testing
- Walk-Forward Analysis

### T3: Web Interface
**Was:** React-basiertes Dashboard
- Agenten-Übersicht
- Shadow-Trading-Status
- Quant-Analytics
- Performance-Charts

### T4: Docker Integration
**Was:** Alles in einem Container
- Multi-Stage Dockerfile
- docker-compose.yml mit allen Services
- Health Checks
- Persistente Volumes

### T5: Tests + Documentation + Push
