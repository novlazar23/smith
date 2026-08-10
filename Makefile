.PHONY: install test lint run compose-up compose-down

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

run:
	uvicorn trading_harness.main:app --reload --port 8080

compose-up:
	docker compose up --build

compose-down:
	docker compose down
