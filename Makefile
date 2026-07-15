.PHONY: install format lint typecheck test run clean docker docker-build

install:
	poetry install

format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

lint:
	poetry run ruff check src/ tests/

typecheck:
	poetry run mypy src/ tests/

test:
	poetry run pytest tests/ -v --cov=src

run:
	poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache outputs multirun wandb htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t arbiter-api:latest .

docker:
	docker-compose up -d
