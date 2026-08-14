# Development Handoff

## Current state

No implementation task is currently in progress. The repository provides a reproducible Python 3.12
environment, a frozen dependency lock, matching local/CI checks, and versioned OpenCode rules and
commands. The complete German OpenCode usage and cross-device workflow is documented in
`docs/opencode-nutzung.md`.

Phase 1 (Research Runtime) additions committed:
- `TradingRun` model with full lifecycle state machine (`RunState`)
- `PerformanceRecord` and `AuditEntry` models
- `TradingRunService` — thread-safe run lifecycle manager with audit trail
- `PerformanceStore` — thread-safe performance records store
- API endpoints: `/runs`, `/runs/{id}`, `/runs/{id}/transition/{state}`, `/runs/{id}/decision`,
  `/runs/{id}/complete`, `/runs/{id}/fail`, `/performance`, `/performance/summary/run/{id}`,
  `/performance/summary/agent/{id}`, `/audit`, `/audit/{entity_id}`
- Policy loader resolves Docker paths (`/app/config/...`) to local paths automatically

Phase 2 (Evaluation) additions committed:
- `MarketRegime` enum — 9 market regimes (strong_bull, weak_bull, range, weak_bear, strong_bear,
  high_volatility, low_volatility, crash, recovery)
- `OutcomeRecord` — actual market outcome for prediction evaluation with MFE/MAE
- `EvaluationResult` — evaluation result storage with metric name and value
- `WalkForwardResult` — per-window walk-forward evaluation result
- `OutcomeGenerator` — generates outcome records from predictions + actual market data,
  stores/retrieves outcomes by agent/run/regime
- `EvaluationService` — comprehensive evaluation metrics engine:
  - Brier Score (probabilistic prediction quality)
  - Expected Calibration Error (ECE) with 5-bin grouping
  - Expectancy (weighted average return per trade)
  - MFE/MAE statistics (avg/max favorable/adverse excursion)
  - Confusion matrix (TP/FP/TN/FN), Precision, Recall, F1
  - Directional Accuracy
  - Per-regime performance evaluation
  - Drawdown calculation (max drawdown, current drawdown, recovery periods)
  - Out-of-Sample evaluation with degradation ratio
  - Walk-Forward stability evaluation with rolling windows
- API endpoints: `/outcomes`, `/outcomes/agent/{id}`, `/outcomes/run/{id}`, `/outcomes/regime/{id}`,
  `/evaluation/agent/{id}`, `/evaluation/regime/{id}/{regime}`,
  `/evaluation/drawdown/{id}`, `/evaluation/out-of-sample`, `/evaluation/walk-forward/{id}`,
  `/evaluation/results`, `/evaluation/results/agent/{id}`

Phase 1 Persistence (PostgreSQL-backed stores) additions committed:
- `Database` (`db.py`) — async PostgreSQL connection pool with schema migration (agents, market_snapshots,
  audit_log, trading_runs, outcomes, evaluation_results), graceful fallback to in-memory on connection failure
- `PersistedAgentRegistry` — PostgreSQL-backed agent registry with in-memory fallback; supports add/list/get/version
- `PersistedSnapshotStore` — PostgreSQL-backed market snapshot store with content-hash (SHA-256) integrity;
  in-memory fallback when DB unavailable
- All stores use `is_available` guard so they never silently swallow errors or block on failed connections
- `datetime` fields correctly parse to `datetime` objects (not strings) from Postgres
- API routes wired to persistent stores; fallback mode works without Postgres running
- Full test suite for persistence: 11 tests covering fallback add/list/get/duplicate/version/hash

## Next priority

Phase 1 remaining: Performance store persistence migration, Outcome Generator persistence migration,
structured agent output queries via API routes (`GET /agent/analyses`, `GET /agent/analyses/run/{run_id}`,
`GET /agent/analyses/agent/{agent_id}`, `GET /agent/analyses/snapshot/{snapshot_id}`).

Phase 2 — Evaluation: Wire EvaluationService results persistence (evaluation_results table exists).

Phase 3 — Evolution: Agent Factory, Genome Persistenz, Mutationen, Recombination,
Challenger Pool, Hall of Fame, Graveyard, Champion/Challenger, Rollback.

Address the security and correctness gaps documented in the README development sequence before
adding paper- or live-trading capabilities. Live execution remains out of scope.

## Resume

```bash
git status -sb
./scripts/bootstrap.sh
make check
opencode
```

Inside OpenCode, run `/resume` to reconstruct context from Git and continue the next safe task.

## Last verification

- `./scripts/bootstrap.sh --check`
- `docker compose config --quiet`
- locked Docker image build and `/health` smoke test
