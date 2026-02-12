.PHONY: install install-dev update uninstall test test-cov lint lint-fix typecheck clean help

PYTHON ?= python3
PIP    ?= pip

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install ideanator (user install from local clone)
	$(PIP) install .

install-dev: ## Install in editable mode with dev dependencies
	$(PIP) install -e ".[dev]"

update: ## Pull latest from GitHub and reinstall
	git pull origin main
	$(PIP) install .

uninstall: ## Uninstall ideanator
	$(PIP) uninstall ideanator -y

test: ## Run the test suite
	$(PYTHON) -m pytest tests/ -v

test-cov: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ --cov=ideanator --cov-report=term-missing

lint: ## Run ruff linter
	$(PYTHON) -m ruff check src/ tests/

lint-fix: ## Run ruff linter with auto-fix
	$(PYTHON) -m ruff check src/ tests/ --fix

typecheck: ## Run mypy type checker
	$(PYTHON) -m mypy src/ideanator/

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
