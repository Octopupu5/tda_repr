.PHONY: install-dev test lint build check-release precommit

install-dev:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check tda_repr tools tests

build:
	python -m build

check-release: build
	python -m twine check dist/*

precommit:
	pre-commit run --all-files

