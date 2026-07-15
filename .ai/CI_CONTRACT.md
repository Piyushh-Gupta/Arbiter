# Purpose

Deterministic Continuous Integration (CI) is required for Arbiter to guarantee that the codebase remains consistently verifiable, reproducible, and resilient across all environments. A strict CI contract prevents configuration drift, surface-level breakages, and inconsistent behavior between developer environments and the authoritative GitHub Actions runner.

# Scope

This contract applies to:

- Local development
- Pull Requests
- GitHub Actions
- Release branches
- Main branch

Every code change must satisfy this contract before integration.



# CI Contract

Every commit submitted to the repository is bound by the CI contract. All changes must satisfy the full validation pipeline locally before they are eligible for commit. Violating the CI contract undermines the integrity of the project's verification layer.

# Canonical Validation Pipeline

Validation must always execute through Poetry to guarantee an isolated and strictly versioned execution context. 

The canonical validation commands are:

```bash
poetry run ruff format --check src/ tests/
poetry run ruff check src/ tests/
poetry run isort --check-only src/ tests/
poetry run mypy src/ tests/
poetry run pytest tests/ --cov=src --cov-report=xml
```

These commands represent the authoritative validation pipeline for the Arbiter project.

# Synchronization Rules

To maintain the integrity of the CI contract, the following rules apply:

- **Strict Parity**: Local validation must mirror GitHub Actions exactly. 
- **Upstream Synchronization**: Any modification to `.github/workflows/ci.yml` must be reflected immediately in this document.
- **Downstream Synchronization**: Any modification to this document requires a synchronized review and update of the GitHub Actions workflow.
- **Zero Drift**: Configuration drift between local validation logic and CI environments is strictly prohibited.

# Failure Policy

- No commit may be created or pushed if any step in the validation pipeline fails.
- CI failures must be comprehensively root-caused before implementation or feature development continues.
- Temporary workarounds, silencing errors without justification, or skipping checks are prohibited.
- Fixes must target the underlying configuration, logic, or environment cause rather than treating symptoms.

# Engineering Principles

The CI contract enforces the following core project principles:

- **Deterministic Builds**: Code formatting, sorting, and type-checking must produce identical outcomes regardless of the machine running them.
- **Reproducible Validation**: Test execution and coverage must be stable and mathematically reproducible.
- **Fail Fast**: The pipeline must reject invalid states at the earliest possible stage.
- **Single Source of Truth**: The pipeline defined in this document and `.github/workflows/ci.yml` is the ultimate arbiter of code quality.
- **Explicit over Implicit Behavior**: Tools must be invoked with explicit configurations rather than relying on global defaults or implicit environment assumptions.
