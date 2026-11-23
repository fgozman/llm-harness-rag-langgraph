.PHONY: help install up down ps logs lint fmt typecheck test test-integration check doctor serve

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install:  ## Create .venv and install all deps
	uv sync

up:  ## Start Qdrant + Phoenix and wait until healthy
	docker compose up -d --wait

down:  ## Stop services (data volumes are kept)
	docker compose down

ps:  ## Show service status
	docker compose ps

logs:  ## Tail service logs
	docker compose logs -f --tail=100

fmt:  ## Format code
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Lint + format check
	uv run ruff check .
	uv run ruff format --check .

typecheck:  ## Strict type check
	uv run pyright

test:  ## Fast unit tests (no services needed)
	uv run pytest

test-integration:  ## Tests needing services/models/APIs
	uv run pytest -m integration

check: lint typecheck test  ## Everything that must be green after each change

doctor:  ## Check service reachability
	uv run rag doctor

serve:  ## Run the HTTP API (POST /ask SSE, POST /ingest, GET /health) on api.host:api.port
	uv run rag serve
