# Git Workflow

## Branching Strategy
- **Trunk-Based Development**: Commits can be pushed directly to `main` for simple refactors and documentation, or small feature branches for major implementations.

## Commit Conventions
- Use lowercase **Conventional Commits**:
  - `feat:` for new capabilities.
  - `fix:` for bug fixes.
  - `chore:` for infrastructure and repository maintenance.
  - `docs:` for documentation updates.
  - `refactor:` for code structure changes without functional impact.
  - `test:` for adding or updating tests.

## Rules
- **No Unapproved Commits**: AI assistants must await user approval before committing or pushing.
- **Atomic Commits**: Keep commits small and logically isolated.
