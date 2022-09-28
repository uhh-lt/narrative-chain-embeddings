include .env
export $(shell sed 's/=.*//' .env)
export TOP_LEVEL_PYTHON_FILES=main.py
export SRC_DIR="eventy"

format:
	venv/bin/python -m black $(SRC_DIR) $(TOP_LEVEL_PYTHON_FILES)
	venv/bin/python -m isort . --profile black

types:
	venv/bin/python -m mypy $(SRC_DIR) $(TOP_LEVEL_PYTHON_FILES)

install-hooks:
	printf "#!/bin/sh\nvenv/bin/python -m black --check $(TOP_LEVEL_PYTHON_FILES) $(SRC_DIR) && venv/bin/python -m isort --profile black --check .\n" > .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit

.PHONY: test
test: venv
	venv/bin/python -m pytest

.PHONY: initdb migrations
