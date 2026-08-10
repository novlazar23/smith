#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap.sh [--check | --docker | --help]

Prepare a fresh Git checkout for development.

  (no option)  Create .env if needed and install the locked development environment
  --check      Bootstrap, then run tests, lint, and type checking
  --docker     Create .env if needed and start the Docker Compose stack
  --help       Show this help without changing local state
EOF
}

ensure_env() {
  if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example"
  fi
}

sync_development_environment() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required for local development: https://docs.astral.sh/uv/" >&2
    exit 1
  fi

  uv sync --frozen --all-extras
}

main() {
  cd "$REPOSITORY_ROOT"

  case "${1:-}" in
    "")
      ensure_env
      sync_development_environment
      ;;
    --check)
      ensure_env
      sync_development_environment
      uv run pytest -q
      uv run ruff check src tests
      uv run mypy src
      ;;
    --docker)
      ensure_env
      docker compose up --build
      ;;
    --help|-h)
      usage
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
