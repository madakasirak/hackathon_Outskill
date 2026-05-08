.PHONY: help install install-dev run run-notebook test test-cov lint format clean

help:  ## Show this help message
	@echo "Deep Researcher — Makefile commands"
	@echo ""
	@echo "  make install       Install runtime dependencies (uses uv)"
	@echo "  make install-dev   Install runtime + dev dependencies"
	@echo "  make run           Launch the Streamlit app"
	@echo "  make run-notebook  Launch Jupyter for the .ipynb"
	@echo "  make test          Run pytest test suite"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make lint          Run ruff linter"
	@echo "  make format        Auto-format with ruff"
	@echo "  make clean         Remove caches, build artifacts, runtime DBs"

install:  ## Install runtime dependencies
	pip install --upgrade uv
	uv pip install --system -e .

install-dev:  ## Install runtime + dev dependencies
	pip install --upgrade uv
	uv pip install --system -e ".[dev]"

run:  ## Launch the Streamlit app
	streamlit run app.py

run-notebook:  ## Launch Jupyter for notebook iteration
	jupyter notebook deep_researcher_skeleton_v3.ipynb

test:  ## Run the test suite
	pytest tests/

test-cov:  ## Run tests with coverage
	pytest tests/ --cov=core --cov-report=term-missing

lint:  ## Lint with ruff
	ruff check .

format:  ## Auto-format with ruff
	ruff check --fix .
	ruff format .

clean:  ## Remove caches and runtime artifacts
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf chroma_db checkpoints.db
	rm -rf *.egg-info dist build
	rm -rf .ipynb_checkpoints
	@echo "Cleaned."
