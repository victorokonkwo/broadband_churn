.PHONY: install install-dev install-api lock lock-upgrade setup-db features train evaluate score serve test lint typecheck clean help

UV      := uv
PYTHON  := uv run python
JUPYTER := uv run jupyter
SRC     := src/churn
PYTEST  := uv run pytest

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────

install:        ## Install runtime + notebook dependencies into .venv
	$(UV) sync --extra notebook

install-dev:    ## Install runtime, notebook, and dev/test dependencies
	$(UV) sync --extra notebook --group dev
	$(UV) run pre-commit install

install-api:    ## Install API serving dependencies
	$(UV) sync --extra api

lock:           ## Regenerate uv.lock from pyproject.toml constraints
	$(UV) lock

lock-upgrade:   ## Upgrade all dependencies and regenerate uv.lock
	$(UV) lock --upgrade

# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

setup-db:       ## Ingest raw data into DuckDB and validate schemas
	$(PYTHON) -m churn.data.loader

features:       ## Build and materialise all feature sets
	$(PYTHON) -m churn.features.feature_store

train:          ## Train LightGBM model with Optuna tuning, log to MLflow
	$(PYTHON) -m churn.training.trainer

evaluate:       ## Run full evaluation suite and save plots to outputs/figures/
	$(PYTHON) -m churn.evaluation.metrics

score:          ## Score all active customers and output scored_customers.csv
	$(PYTHON) -m churn.scoring.batch_scorer

explain:        ## Generate SHAP global + local explainability plots
	$(PYTHON) -m churn.explainability.shap_explainer

monitor:        ## Run Evidently drift report against latest scored batch
	$(PYTHON) -m churn.monitoring.drift_detector

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────

serve:          ## Start FastAPI server (dev mode with reload)
	$(UV) run uvicorn churn.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod:     ## Start FastAPI server (production mode)
	$(UV) run uvicorn churn.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# ─────────────────────────────────────────────
# Quality
# ─────────────────────────────────────────────

test:           ## Run all tests with coverage
	$(PYTEST) --cov=$(SRC) --cov-report=term-missing --cov-report=html

test-unit:      ## Run unit tests only
	$(PYTEST) tests/unit -v

test-integration: ## Run integration tests only
	$(PYTEST) tests/integration -v -m integration

test-leakage:   ## Run data leakage checks
	$(PYTEST) tests/data_quality -v

lint:           ## Lint and auto-fix with ruff
	$(UV) run ruff check $(SRC) tests --fix
	$(UV) run ruff format $(SRC) tests

typecheck:      ## Type-check with mypy
	$(UV) run mypy $(SRC)

# ─────────────────────────────────────────────
# Docker
# ─────────────────────────────────────────────

docker-build-api:   ## Build the API Docker image
	docker build -f docker/Dockerfile.api -t churn-api:latest .

docker-build-train: ## Build the training pipeline Docker image
	docker build -f docker/Dockerfile.training -t churn-training:latest .

docker-up:      ## Start local stack (API + MLflow) with docker-compose
	docker compose -f docker/docker-compose.yml up -d

docker-down:    ## Stop local stack
	docker compose -f docker/docker-compose.yml down

# ─────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────

reports:        ## Export all notebooks to PDF in outputs/reports/
	mkdir -p outputs/reports
	$(JUPYTER) nbconvert --to webpdf --output-dir outputs/reports notebooks/*.ipynb

reports-clean:  ## Export all notebooks to PDF (no code cells — stakeholder view)
	mkdir -p outputs/reports
	$(JUPYTER) nbconvert --to webpdf --no-input --output-dir outputs/reports notebooks/*.ipynb

install-browsers: ## Install Chromium for PDF export (run once after uv sync)
	$(UV) run playwright install chromium

# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

clean:          ## Remove all generated artefacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache htmlcov .coverage
	rm -rf outputs/figures/* outputs/reports/*

help:           ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
