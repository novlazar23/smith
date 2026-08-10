.PHONY: bootstrap sync check test lint typecheck run compose-up compose-down

bootstrap:
	./scripts/bootstrap.sh

sync:
	uv sync --frozen --all-extras

check: test lint typecheck

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

run:
	uv run uvicorn trading_harness.main:app --reload --port 8080

compose-up:
	docker compose up --build

compose-down:
	docker compose down
