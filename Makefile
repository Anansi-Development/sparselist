.PHONY: help venv install build test check lint format type clean docs publish

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

venv:  ## Create venv and install all dev dependencies
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements-dev.txt
	./venv/bin/pip install -e .
	@echo "Development environment ready. Activate with: source venv/bin/activate"

install:  ## Install package in development mode
	pip install -e .

build:  ## Build distribution packages (wheel and source)
	python3 -m build

test:  ## Run tests with pytest (accepts args: make test ARGS="-k test_name")
	pytest $(ARGS)

test-all:  ## Run tests across all supported Python versions using tox
	tox

check:  ## Run all checks (lint, format check, type check)
	@echo "Running ruff linting..."
	ruff check src/sparselist tests
	@echo "Running ruff format check..."
	ruff format --check src/sparselist tests
	@echo "Running mypy type checking..."
	mypy src/sparselist

lint:  ## Run ruff linter
	ruff check src/sparselist tests

format:  ## Format code with ruff
	ruff format src/sparselist tests
	ruff check --fix src/sparselist tests

type:  ## Run mypy type checker
	mypy src/sparselist

coverage:  ## Run tests with coverage report
	pytest --cov --cov-report=term-missing --cov-report=html

docs: install  ## Clean and rebuild Sphinx documentation
	@echo "Cleaning previous documentation build..."
	cd docs && $(MAKE) clean
	@echo "Building documentation..."
	cd docs && $(MAKE) html
	@echo "Documentation built. Open docs/_build/html/index.html"

clean:  ## Remove build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf src/*.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .tox/
	rm -rf htmlcov/
	rm -rf docs/_build/
	rm -f .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

publish-test:  ## Build and publish to TestPyPI (placeholder for future)
	@echo "TestPyPI publishing not yet configured"
	@echo "Run: twine upload --repository testpypi dist/*"

publish:  ## Build and publish to PyPI (placeholder for future)
	@echo "PyPI publishing not yet configured"
	@echo "Run: twine upload dist/*"
