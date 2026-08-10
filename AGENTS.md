# Repository Guidelines

## Mission and status

This repository is a shadow-first evolutionary trading research service. It is an early MVP, not
a production trading system. Keep generation, evaluation, promotion, trading decisions, and
execution technically separated. Live-Execution must remain disabled until the documented safety,
evaluation, and audit requirements are implemented and verified.

## Start or resume work

1. Run `git status -sb` and inspect the current branch, recent commits, and pending diff.
2. Read `docs/handoff.md`, then the relevant README and architecture sections.
3. On a fresh checkout run `./scripts/bootstrap.sh`; otherwise run `uv sync --frozen --all-extras`.
4. Continue the active handoff task. If none exists, select the highest-priority safe item from the
   README development sequence and state the assumption before implementing it.

The repository is the complete shared context. Do not depend on a specific coding agent, an
external harness, conversation memory, or files outside this worktree.

## Development workflow

- Use Python 3.12, type annotations, Pydantic schemas, and small testable services.
- For behavior changes, write a failing regression test first, implement the smallest fix, then
  refactor with the test still green.
- Keep business logic out of HTTP routes.
- Update documentation and `docs/handoff.md` when behavior, setup, architecture, or task state
  changes.
- Before declaring work complete run `make check`. For container changes also run
  `docker compose config --quiet` and build the image.
- Report actual verification output. Do not call placeholders, TODOs, or unverified behavior done.

## Safety boundaries

- Never commit or expose `.env`, credentials, API keys, private certificates, database dumps, or
  exchange secrets. Use `.env.example` only for non-secret templates.
- Never enable Live-Execution or weaken the kill switch, deterministic risk limits, symbol allow
  list, leverage limits, order deduplication, or execution isolation without explicit user approval.
- LLM output may propose or explain trades but must never override deterministic risk policy.
- Do not perform destructive Git operations, delete persistent volumes, push, merge, or publish
  releases without explicit user approval.
- Preserve unrelated worktree changes.

## Project commands

- Bootstrap: `./scripts/bootstrap.sh`
- Full verification: `make check`
- Run API: `make run`
- Docker stack: `./scripts/bootstrap.sh --docker`
- OpenCode helpers: `/resume`, `/check`, `/handoff`

Use feature branches and Conventional Commit messages. A handoff is complete only when the relevant
changes, tests, documentation, and current `docs/handoff.md` are committed to Git.
