# EPIC-17: Production Hardening — Infrastructure, Security, DR

## Problem

Das System ist funktional vollständig (EPIC-01 bis EPIC-16), aber noch nicht production-ready. Die Infrastruktur ist auf Docker Compose beschränkt, Sicherheitsmechanismen sind teilweise implementiert, Disaster-Pläne existieren nicht, Performance-Benchmarks fehlen und Compliance-Anforderungen (MiFID II, GDPR, SOC 2) sind nicht adressiert.

Laut Spezifikation (Abschnitt 26, 27, 29) sind vollständige Sicherheit, Observability und Teststrategie erforderlich, aber nur teilweise umgesetzt.

## Ziel

Das System in einen production-ready Zustand überführen mit:
- Kubernetes-bereiter Multi-Region-Infrastruktur
- Vollständigem RBAC und Security Hardening
- Getesteten Disaster Recovery Prozessen
- Performance-Zielen (<500ms API, skalierbare Ingestion)
- Compliance-Readiness (MiFID II, GDPR, SOC 2)

## Arbeitspakete

### WP01: Production Infrastructure

#### Beschreibung
Migration von Docker Compose → Kubernetes (Helm Charts), Multi-Region-Deployment, Datenbank-Replication, Redpanda mit Replikation, Auto-Scaling.

#### Implementierung
- `infrastructure/k8s/` — Helm Charts für alle Services
  - `charts/trading-orchestrator/Chart.yaml`
  - `charts/trading-orchestrator/values.yaml`
  - `charts/trading-orchestrator/templates/` — Deployments, Services, HPA, Ingress
- `infrastructure/k8s/namespace.yaml` — Multi-Environment Namespaces (staging, production)
- `infrastructure/k8s/cert-manager.yaml` — TLS-Zertifikate automation
- `infrastructure/k8s/priority-classes.yaml` — Pod Priority für kritische Services
- `packages/persistence/postgresql/replication.py` — Streaming Replication Config
- `packages/persistence/clickhouse/replication.py` — ClickHouse Distributed Tables
- `packages/streaming/redpanda/config.py` — Replication Factor, ISR Settings
- `configs/k8s/staging.yaml` — Staging Configuration
- `configs/k8s/production.yaml` — Production Configuration

#### Akzeptanzkriterien
- [ ] Helm Chart deployed `helm install trading-orchestrator charts/trading-orchestrator` → alle Pods Running
- [ ] HPA konfiguriert: CPU > 70% → Scale Up, Min 2 Replicas pro Service
- [ ] PostgreSQL streaming replication: primary + 1 replica, failover < 30s
- [ ] ClickHouse replication: 2-shard 2-replica Cluster
- [ ] Redpanda Topics: replication_factor=3, min_isr=2
- [ ] Multi-Region: EU-primary + US-failover, DNS-basierter Routing

### WP02: Security Hardening

#### Beschreibung
Vollständiges RBAC, API-Key-Vault, Network Segmentation, Audit-Logging, Dependency Scanning.

#### Implementierung
- `packages/security/rbac.py` — Role-Based Access Control Engine
  - `SecurityRole` enum: viewer, researcher, operator, risk_manager, administrator, auditor
  - `Permission` model: resource, action, condition
  - `RBACMiddleware` class: per-request role check
- `packages/security/vault.py` — Secret Vault Integration
  - `SecretBackend` protocol: HashiCorp Vault, AWS Secrets Manager, Local
  - `encrypt_api_key(plaintext) → ciphertext` (AES-256-GCM)
  - `decrypt_api_key(ciphertext) → plaintext`
  - `rotate_api_key(old_key_id) → new_key_id`
- `packages/security/network_zones.py` — Network Segmentation
  - `ingestion_zone`, `analysis_zone`, `execution_zone`, `admin_zone`
  - Inter-Zone Firewall Rules als konfigurierbare Policy
- `packages/security/audit.py` — Immutable Audit Trail
  - `AuditEvent` schema: event_id, actor, action, resource, before/after, timestamp, ip
  - Append-only storage (PostgreSQL with HASH verification)
- `scripts/security/snyk_scan.py` — Dependency Vulnerability Scanning
- `scripts/security/semgrep_scan.py` — Static Analysis Security Testing
- `apps/api/middleware/auth.py` — JWT/Bearer Token + Role Validation

#### Akzeptanzkriterien
- [ ] Alle 6 RBAC-Rollen implementiert mit expliziten Permission-Checks
- [ ] API-Keys AES-256-GCM verschlüsselt, rotation unterstützt
- [ ] Network Zones implementiert mit Inter-Zone Rules
- [ ] Audit Trail append-only, HASH-Verifikation pro Eintrag
- [ ] snyk scan → 0 critical, 0 high vulnerabilities
- [ ] semgrep scan → 0 critical, 0 high issues

### WP03: Disaster Recovery

#### Beschreibung
Point-in-Time Recovery, Runbooks, RTO/RPO-Ziele, Backup-Verifikation.

#### Implementierung
- `packages/persistence/postgresql/wal_archiver.py` — WAL-Archiving to S3
  - `start_wal_archive() → None` — konfiguriert pg_basebackup + WAL-Archivierung
  - `restore_from_wal(target_time: datetime) → None` — PITR Restore
- `packages/persistence/clickhouse/snapshot.py` — ClickHouse Snapshot Export
  - `create_snapshot(table_pattern: str) → snapshot_id`
  - `restore_snapshot(snapshot_id: str) → None`
  - `export_to_s3(snapshot_id: str, bucket: str) → None`
- `scripts/dr/runbook_exchange_key_compromise.sh` — Runbook: Exchange API Key Kompromittierung
  -立即 Key Rotation (Vault)
  - Cancel all open orders
  - Move to paper mode
  - Security team notification
- `scripts/dr/runbook_market_crash.sh` — Runbook: Markt-Crash Kill-Switch
  - Trigger system-wide kill-switch
  - Cancel all positions
  - Switch to paper mode
  - Generate incident report
- `scripts/dr/runbook_region_outage.sh` — Runbook: Cloud-Region Ausfall
  - DNS failover zu Backup-Region
  - Database failover
  - Service reconstitution
- `scripts/dr/backup_verify.py` — Automatisierte Backup-Verifikation
  - Restore Backup in Test-DB
  - Compare row counts
  - Compare sample records
- `docs/runbooks/` — Vollständige Runbook-Dokumentation

#### Akzeptanzkriterien
- [ ] WAL-Archiving aktiv, Restore getestet mit < 1h RPO
- [ ] ClickHouse Snapshots exportiert und restauriert
- [ ] Alle Runbooks dokumentiert und getestet
- [ ] Backup-Verifikation automatisiert, wöchentlich ausgeführt
- [ ] RTO < 15 Minuten für kritische Services definiert
- [ ] RPO < 5 Minuten für alle Datenbanken

### WP04: Performance Engineering

#### Beschreibung
Lasttests, Memory Profiling, Cold-Start-Optimierung, Datenbank-Optimierung, API-Budgets.

#### Implementierung
- `tests/performance/test_ingestion_load.py` — Ingestion Load Test
  - 1000 candles/sec sustained
  - Duration: 10 minutes
  - Memory usage tracked
- `tests/performance/test_analysis_throughput.py` — Analysis Throughput Test
  - 100 analysis runs/sec concurrent
  - Latency P50, P95, P99 tracked
- `scripts/profile/memory_leak_check.py` — Memory Leak Profiler
  - Profile over 24h operation
  - Alert on monotonic memory growth
  - Compare GC collections
- `scripts/optimize/cold_start.py` — Cold Start Benchmark & Optimization
  - Measure container startup time
  - Profile Python import overhead
  - Pre-compile hot paths
- `packages/persistence/postgresql/index_optimizer.py` — Index Recommendation
  - Analyze slow queries (PostgreSQL pg_stat_statements)
  - Recommend composite indexes
  - Monitor index usage
- `packages/persistence/postgresql/connection_pool.py` — Connection Pool Tuning
  - Dynamic pool sizing based on load
  - Idle connection cleanup
  - Stale connection detection
- `apps/api/middleware/rate_limiter.py` — API Rate Limiting
  - Per-client rate limits
  - Sliding window algorithm
  - Configurable per-endpoint

#### Akzeptanzkriterien
- [ ] 1000 candles/sec ingestion → < 100ms processing latency
- [ ] 100 analysis runs/sec → P95 < 5s
- [ ] Health endpoint < 500ms unter Last
- [ ] Analysis creation < 5s unter Last
- [ ] Memory growth < 5% over 24h steady state
- [ ] Database queries: < 100ms P95, recommended indexes applied

### WP05: Compliance & Audit

#### Beschreibung
Trade Log Immütlichkeit, Regulatory Reporting, Data Retention, GDPR, SOC 2 Readiness.

#### Implementierung
- `packages/compliance/trade_log.py` — Immutable Trade Log
  - `TradeLogEntry` with cryptographic chain (SHA-256 hash chain)
  - `verify_integrity() → bool` — chain verification
  - Append-only PostgreSQL table
- `packages/compliance/mifid2.py` — MiFID II Reporting Templates
  - `generate_transaction_report(order) → Mifid2Report`
  - `generate_best_execution_report(venue_comparison) → BestExecutionReport`
  - `generate_position_report(positions) → PositionReport`
- `packages/compliance/gdpr.py` — GDPR Compliance
  - `right_to_erasure(user_id) → None` — Erase non-trade data
  - `data_export(user_id) → ExportArchive` — Data portability
  - Retention policy engine
- `scripts/compliance/soc2_readiness.py` — SOC 2 Type I Checklist
  - Access control documentation
  - Change management audit trail
  - Incident response procedure
  - Business continuity documentation
- `packages/compliance/retention.py` — Data Retention Policies
  - `RetentionRule` model: data_type, retention_period, action
  - Automatic archival/deletion based on policy
  - Audit trail for all deletions

#### Akzeptanzkriterien
- [ ] Trade Log: hash chain verifiziert, integrity check 100%
- [ ] MiFID II Reports generiert und format-korrekt
- [ ] GDPR Right-to-Erasure getestet und dokumentiert
- [ ] GDPR Data Export funktioniert (CSV/JSON)
- [ ] Retention Policies aktiv und automatisch angewendet
- [ ] SOC 2 Type I Checklist completed with findings documented

## Kritischer Pfad

`WP02 → WP04` (Sicherheit vor Performance)
`WP03 → WP05` (DR vor Compliance)
`WP01 → WP04` (Infrastruktur vor Performance)

## Parallele Gruppen

- **P1**: WP01 (Infrastruktur) — kann parallel zu allen anderen laufen
- **P2**: WP02 (Sicherheit) + WP05 (Compliance) — beide security/compliance fokussiert
- **P3**: WP03 (DR) + WP04 (Performance) — beide test-basiert

## Risiken

1. **Kubernetes-Komplexität** — Mitbringen: Helm Charts mit Helmfile, automatisierte Tests im CI
2. **Multi-Region Latenz** — Mitbringen: Region-spezifische DB-Endpoints, read-replicas lokal
3. **Compliance-Berater** — Mitbringen: Templates aus offenen Frameworks (SOC 2 Ready), Review durch externen Auditor

## Rollback-Strategie

- Docker Compose bleibt als Fallback (configs/staging.yaml, configs/trading.yaml)
- Alle Änderungen versioniert in Git, reversible Commits
- Kubernetes-Deployment mit Blue/Green: `kubectl rollout undo`