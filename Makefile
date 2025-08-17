.PHONY: help install install-dev test lint format type-check pre-commit clean build publish

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install the package"
	@echo "  install-dev   Install the package with development dependencies"
	@echo "  test          Run tests with coverage"
	@echo "  lint          Run linting with ruff"
	@echo "  format        Format code with ruff"
	@echo "  type-check    Run type checking with mypy"
	@echo "  pre-commit    Run pre-commit hooks"
	@echo "  clean         Clean build artifacts"
	@echo "  build         Build the package"
	@echo "  publish       Publish to PyPI"

# Installation
install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

# Testing
test:
	uv run python -m pytest

# Code quality
lint:
	ruff check .

format:
	ruff format .

type-check:
	mypy src/pydantic_ai_optimizers/

# Pre-commit
pre-commit:
	pre-commit run --all-files

# Build and publish
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	uv build

publish: build
	uv publish

# Development workflow
dev-setup: install-dev
	pre-commit install

check: lint type-check test

all: format lint type-check test