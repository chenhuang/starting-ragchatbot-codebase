- Always use uv to execute.

## Code Quality Tools
- Run quality checks: `uv run python scripts/quality_check.py`
- Format code: `uv run python scripts/format_code.py`
- Individual tools:
  - Black formatting: `uv run black .`
  - Import sorting: `uv run isort .`  
  - Linting: `uv run flake8 .`
  - Type checking: `uv run mypy backend/`