.PHONY: help install run test protocols-check test-unit test-property test-integration test-coverage coverage-report coverage-gaps clean format lint

help:
	@echo "APGI REST API - Available commands:"
	@echo "  make install           - Install dependencies"
	@echo "  make run               - Run the API server"
	@echo "  make test              - Run all tests"
	@echo "  make protocols-check   - Validate the 1.1.0 protocol specification"
	@echo "  make test-unit         - Run unit tests only"
	@echo "  make test-property     - Run property-based tests only"
	@echo "  make test-integration  - Run integration tests only"
	@echo "  make test-coverage     - Run tests with coverage analysis"
	@echo "  make coverage-report   - Generate and display coverage report"
	@echo "  make coverage-gaps     - Analyze coverage gaps in detail"
	@echo "  make clean             - Clean up generated files"
	@echo "  make format            - Format code with black and isort"
	@echo "  make lint              - Run linting checks"

install:
	pip install -r requirements.txt

run:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

protocols-check:
	python3 scripts/validate_protocols.py

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m unit

test-property:
	pytest tests/property/ -v -m property

test-integration:
	pytest tests/integration/ -v -m integration

test-coverage:
	python3 utils/run_coverage.py

coverage-report:
	pytest tests/ -v --cov=apgi_framework --cov=api --cov-report=html --cov-report=term-missing
	@echo "\nHTML report: htmlcov/index.html"

coverage-gaps:
	python3 utils/analyze_gaps.py

clean:
	python3 delete_pycache.py --yes
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.json" -delete
	find . -type f -name "coverage.xml" -delete
	find . -type f -name "coverage_gap_analysis.json" -delete

format:
	black apgi_framework/ api/ tests/
	isort apgi_framework/ api/ tests/

lint:
	flake8 apgi_framework/ api/ tests/
	mypy apgi_framework/ api/ --ignore-missing-imports
