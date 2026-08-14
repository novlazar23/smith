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
  audit_log, trading_runs, outcomes, evaluation_results, performance_records), graceful fallback to in-memory on connection failure
- `PersistedAgentRegistry` — PostgreSQL-backed agent registry with in-memory fallback; supports add/list/get/version
- `PersistedSnapshotStore` — PostgreSQL-backed market snapshot store with content-hash (SHA-256) integrity;
  in-memory fallback when DB unavailable
- `PersistedPerformanceStore` — PostgreSQL-backed performance records store with in-memory fallback;
  supports add/get/all/by_run/by_agent/by_snapshot and upsert-on-conflict
- All stores use `is_available` guard so they never silently swallow errors or block on failed connections
- `datetime` fields correctly parse to `datetime` objects (not strings) from Postgres
- API routes wired to persistent stores; fallback mode works without Postgres running
- Performance store test suite: 11 tests covering fallback add/get/all filter methods, upsert/overwrite, and defaults
- `GET /performance/summary/snapshot/{snapshot_id}` endpoint added to routes

## Next priority

Phase 1 remaining:
- **Outcome Generator persistence migration** — `OutcomeGenerator` currently only stores in-memory; needs PostgreSQL-backed store using the existing `outcomes` table
- **Structured agent output queries via API routes** — expose existing `PersistedAgentAnalysisStore` data through endpoints (`GET /agent/analyses`, `GET /agent/analyses/run/{run_id}`, `GET /agent/analyses/agent/{agent_id}`, `GET /agent/analyses/snapshot/{snapshot_id}`)

Phase 2 — Evaluation: ✅ EvaluationService results persistence wired — `PersistedEvaluationResultStore` added, injected into `EvaluationService` via `routes.py`, `add()` called after every `evaluate_agent()`, test suite with 8 tests, `make check` clean.

Phase 3 — Evolution: ✅ COMPLETE. 8 core services, 6 test files, 24 API endpoints wired.

Services:
- `AgentGenomeStore` — in-memory genome persistence (CRUD, filtering by category/status/generation)
- `PersistedAgentGenomeStore` — PostgreSQL-backed variant with schema migration
- `AgentFactory` — mutation strategies (INDICATOR_ADD, INDICATOR_REMOVE, TIMEFRAME_ADD,
  TIMEFRAME_REMOVE, PARAMETER_MUTATION, TEMPERATURE_MUTATION, RECOMBINATION, SPECIALIZATION,
  SIMPLIFICATION, DIVERSITY_INJECTION)
- `ChallengerPool` — champion/challenger pairing, promotion evaluation, promotion execution,
  demotion, category stats
- `HallOfFame` — top-performer tracking with max-entry limit, category filtering, best lookup
- `Graveyard` — retired/rejected agent tracking with category filtering
- `EvolutionService` — orchestrator tying factory + pool + hall-of-fame + graveyard + rollback
- `PromotionPolicy` — minimum_observations >= 10, relative_improvement_min >= 0.03,
  out-of-sample, walk-forward, shadow mode, ensemble contribution, security gates

Models added to `models.py`:
- `MutationType` enum (10 mutation types)
- `GenomeMutation` — mutation record with type, changes, and hypothesis
- `ChampionChallenger` — pair record for evaluation
- `EvolutionRun` — promotion/retirement run record
- `PromotionDecision` — decision with approved/rejected and reasons
- `RollbackEntry` — status transition audit record
- `HallOfFameRecord` / `GraveyardRecord` — lifecycle archives

Test suites (1193 lines total):
- `test_agent_genome_store.py` — 15 tests (CRUD, filtering, hash)
- `test_agent_factory.py` — 10+ tests (all mutation strategies, recombination)
- `test_challenger_pool.py` — 10+ tests (pairing, promotion eval, demotion, stats)
- `test_hall_of_fame.py` — 9 tests (add, sorting, limits, lookups)
- `test_graveyard.py` — 6 tests (add, retrieve, category filter)
- `test_evolution_service.py` — 18+ tests (mutant generation, recombination, challenger mgmt,
  promotion decisions, retire/reject, probation, rollback)

API endpoints added to `routes.py` (24 new endpoints):
- `POST /evolution/mutate` — generate mutant from parent
- `POST /evolution/recombine` — recombine two parents
- `POST /evolution/challengers/{agent_id}/add` — add challenger to pool
- `GET /evolution/challengers/pairs/{category}` — list champion/challenger pairs
- `POST /evolution/challengers/evaluate` — evaluate promotion criteria
- `POST /evolution/challengers/promote` — execute promotion
- `POST /evolution/challengers/demote` — demote to probation
- `POST /evolution/hall-of-fame` — add to hall of fame
- `GET /evolution/hall-of-fame` — list all hall of fame records
- `GET /evolution/hall-of-fame/{category}` — list by category
- `GET /evolution/hall-of-fame/top/{category}` — get best in category
- `POST /evolution/graveyard` — add to graveyard
- `GET /evolution/graveyard` — list all graveyard records
- `GET /evolution/graveyard/{category}` — list by category
- `GET /evolution/promotion-history/{category}` — list promotion runs
- `GET /evolution/rollbacks` — list all rollbacks
- `GET /evolution/rollbacks/{agent_id}` — list rollbacks for agent
- `POST /evolution/rollback` — rollback agent status
- `GET /evolution/population-stats/{category}` — category population stats
- `GET /evolution/population-stats` — all category stats

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
