# Development Workflow

## The AI-Assisted Lifecycle
All development follows a strict Planning → Implementation → Review → Commit lifecycle.

1. **Planning Mode**: Before executing complex implementations, produce an Implementation Plan outlining proposed changes.
2. **Implementation**: Write the application logic, configuration, and tests.
3. **Review & Validation**: Verify type safety (`make typecheck`), formatting (`make format`), linting (`make lint`), and tests (`make test`).
4. **Approval Request**: Stop and await architectural approval from the user. **Never commit without explicit instruction.**
5. **Commit**: Once approved, execute Git operations using Conventional Commits.
