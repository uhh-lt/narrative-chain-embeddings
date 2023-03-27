export TOP_LEVEL_PYTHON_FILES=main.py t5_data.py
export SRC_DIR="eventy"

format:
	python -m black $(SRC_DIR) $(TOP_LEVEL_PYTHON_FILES)
	python -m isort . --profile black

types:
	python -m mypy $(SRC_DIR) $(TOP_LEVEL_PYTHON_FILES)

install-hooks:
	printf "#!/bin/sh\npython -m black --check $(TOP_LEVEL_PYTHON_FILES) $(SRC_DIR) && python -m isort --profile black --check .\n" > .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit

.PHONY: test
test: venv
	python -m pytest

.PHONY: initdb migrations
