# EPIC-13: Production Readiness — CI/CD, Deployment & Hardening

## Problem
EPIC-01 bis EPIC-12 haben das komplette Trading-System gebaut. Die Infrastruktur läuft lokal über Docker Compose. Es fehlen:
- CI/CD-Pipeline für automatisierte Tests, Linting, Security-Scans
- Production-Deployment-Scripte mit Configuration Management
- Docker-Images mit Multi-Stage-Build, Health Checks, Resource Limits
- Environment Configuration für Staging und Production
- Infrastructure as Code (docker-compose.prod.yaml mit TLS, Rate Limiting, Health Checks)
- Load/Stress Testing für die API und Event-Pipeline

## Ziel
Automatisierte Release-Pipeline und Production-ready Deployment-Konfiguration mit:
- CI/CD Pipeline (GitHub Actions) mit Tests, ruff, security scan, docker build
- Multi-Stage Docker Builds für alle Services
- Production Docker Compose mit TLS, Health Checks, Resource Limits
- Staging- und Production-Environment-Konfiguration
- Load Testing für API-Endpunkte und Event-Pipeline
- Infrastructure Hardening (secrets management, logging, monitoring)

## Abhängigkeiten
- EPIC-01 bis EPIC-12 (alle vorangehenden Epics)

## Arbeitspakete

### WP01: CI/CD Pipeline (GitHub Actions)
- .github/workflows/ci.yml — Build, Test, Lint, Security Scan
- .github/workflows/release.yml — Docker Image Build & Push
- .github/workflows/staging.yml — Staging Deployment Test
- Steps: checkout, setup-python, ruff check, pytest (all), docker build --check, trivy scan
- Cache für Python dependencies
- Parallel Matrix: Python 3.12, Tests parallel
- File: .github/workflows/*.yml

### WP02: Multi-Stage Docker Builds
- apps/api/Dockerfile.prod — Multi-Stage mit Alpine
- packages/observability/Dockerfile.prod — Multi-Stage mit Alpine
- Health Checks in Dockerfiles (HEALTHCHECK directive)
- Resource Limits in docker-compose (deploy.resources)
- .dockerignore — exclude test, venv, __pycache__, .git
- File: apps/api/Dockerfile.prod, packages/observability/Dockerfile.prod, .dockerignore

### WP03: Production Docker Compose
- infrastructure/docker-compose.prod.yaml — Production-Konfiguration
- TLS/Terminierung via Nginx (Existing nginx config anpassen für Production)
- Health Checks für alle Services (Postgres, ClickHouse, Redpanda, Redis, MinIO)
- Resource Limits (memory, CPU) für alle Services
- Volume Mounts für Production (persistent data)
- Network Policies (isolation trading-network, monitoring-network)
- File: infrastructure/docker-compose.prod.yaml

### WP04: Environment Configuration
- configs/staging.yaml — Staging-Configuration (reduced risk limits, test data)
- configs/production.yaml — Production-Configuration
- Environment Variable Documentation (.env.example)
- Secret Rotation Guide
- File: configs/staging.yaml, configs/production.yaml, .env.example

### WP05: Load/Stress Testing
- tests/load/test_api_load.py — API Endpunkt Load Tests
- tests/load/test_event_pipeline_stress.py — Event Pipeline Stress Tests
- Locust oder custom stress test framework
- Benchmark Results als Evidence
- File: tests/load/*.py

### WP06: Infrastructure Hardening
- Secret Management: ENV vars in Production, no hardcoded values
- Structured Logging mit Log-Rotation in Production
- Prometheus Alerting Rules für kritische Metriken
- Grafana Dashboard für Production Monitoring
- File: infrastructure/prometheus-alerts.yml, infrastructure/grafana-dashboards/*.json

## DoD
- CI Pipeline läuft für jeden Commit und PR
- Docker Images builden ohne Secrets
- Production Docker Compose startet alle Services ohne Fehler
- Health Checks prüfen alle Services
- Load Test: API hält 100 req/s, Event-Pipeline 10k msg/s
- Alle Tests bestanden, ruff clean
- Environment-Konfigurationen validiert

## Risiken
- CI-Pipeline zu langsam (>30min)
- Docker Images zu groß (>1GB)
- Resource Limits nicht optimiert

## Rollback
- Ältere Docker Images über Registry verfügbar halten
- docker-compose.prod.yaml versioniert, Rollback via git checkout