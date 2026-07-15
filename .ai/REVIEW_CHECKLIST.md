# Review Checklist

Every implementation must pass this checklist before approval and commit.

## 1. Architecture Compliance
- [ ] Does this change respect the monolithic, single-loop pipeline?
- [ ] Are all dependencies within the approved `TECH_STACK.md`?

## 2. Code Quality
- [ ] Does `make typecheck` (mypy) pass?
- [ ] Does `make lint` and `make format` (ruff) pass?

## 3. Testing
- [ ] Does `make test` pass?
- [ ] Are new features covered by unit or integration tests?

## 4. Documentation
- [ ] Are functions and classes documented concisely?
- [ ] Did this change require an update to the `DECISION_LOG.md`?

## 5. Security & Maintainability
- [ ] Are secrets excluded from source code and managed via `.env`?
- [ ] Is logging implemented via `structlog`?

## 6. Root Cause Validation
- [ ] Was the actual root cause identified?
- [ ] Is every modification directly related to that root cause?
- [ ] Were unrelated files left untouched?
