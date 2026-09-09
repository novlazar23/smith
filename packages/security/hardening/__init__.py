"""Security Hardening für Live Trading (EPIC-16 WP06).

Module:
- ``encryption`` — AES-256-GCM KeyRing für API-Keys at rest
- ``secret_rotation`` — Rotations-Policy + zero-downtime Key-Rotation
- ``audit_live`` — append-only Hash-Chain-Audit-Trail für Live-Operationen
- ``rbac_live`` — RBAC-Dependency für Live-API-Endpunkte
- ``ip_whitelist`` — IP-Allowlist + Middleware für Live-Endpunkte
- ``api_rate_limiter`` — per-IP Sliding-Window-Rate-Limiting für Live-Endpunkte

Dieses Package importiert bewusst keine optionalen Abhängigkeiten auf
Modul-Ebene; ``cryptography`` wird lazily innerhalb von
:mod:`encryption` geladen.
"""
