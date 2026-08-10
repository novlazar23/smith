# Development Handoff

## Current state

No implementation task is currently in progress. The repository provides a reproducible Python 3.12
environment, a frozen dependency lock, matching local/CI checks, and versioned OpenCode rules and
commands. The complete German OpenCode usage and cross-device workflow is documented in
`docs/opencode-nutzung.md`.

## Next priority

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
