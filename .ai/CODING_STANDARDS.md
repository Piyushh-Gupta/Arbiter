# Coding Standards

## Python Conventions
- **Version**: Python 3.12.
- **Typing**: Strict type hinting is mandatory. Enforced via `mypy`.
- **Formatting**: `ruff format` is the sole standard.
- **Linting**: `ruff check` is mandatory. No unused imports or variables.

## Engineering Practices
- **Configuration**: Managed via `pydantic-settings` and `Hydra`. No hardcoded magic strings.
- **Logging**: Use `structlog` for structured JSON logging. Every log must carry contextual metadata.
- **Testing**: `pytest` is standard. High coverage is expected for application logic.
- **Dependencies**: Poetry is the sole package manager. Avoid introducing new dependencies without architectural approval.

See `TECH_STACK.md` for approved frameworks.
