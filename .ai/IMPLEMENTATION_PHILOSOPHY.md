# Implementation Philosophy

This document outlines the core engineering philosophy for the Arbiter project. While `ARCHITECTURE_RULES.md` and `CODING_STANDARDS.md` enforce explicit constraints, these principles guide engineering judgement and decision-making during implementation.

## 1. Simplicity and Explicitness
- **Simplicity over Complexity**: Always choose the simplest architectural or algorithmic solution that satisfies the research and functional requirements.
- **Readability over Cleverness**: Code is read vastly more often than it is written. Optimize for immediate human comprehension.
- **Explicitness over Implicit Behaviour**: Avoid "magic" frameworks, implicit global state, or hidden side effects. Control flows and data transformations must be transparent.

## 2. Robustness and Reliability
- **Reproducibility**: Treat reproducibility as a first-class requirement. Guarantee deterministic behaviour via strict random seeding and frozen environments over developer convenience.
- **Fail Fast**: Validate inputs early and crash aggressively with meaningful, actionable error messages rather than silently passing malformed state.
- **Security by Default**: Never trust input. Manage secrets exclusively via environments, and assume all external data is hostile.
- **Stable Milestones**: Every implementation milestone must leave the repository in a deployable, tested, and verifiable state.

## 3. Architecture and Design
- **Single Responsibility Principle**: Functions, classes, and modules should have exactly one reason to change.
- **Modular, Loosely Coupled Components**: Isolate logic behind clear interfaces to ensure every implementation is reviewable and testable in isolation.
- **Clear Separation of Concerns**: Maintain strict boundaries between infrastructure, routing, verification, and data persistence layers.
- **Configuration over Hardcoding**: Extract all mutable parameters, paths, and thresholds into Hydra configuration files or environment variables.
- **Minimal Dependencies**: Do not introduce new libraries unless the value overwhelmingly justifies the maintenance burden.

## 4. Engineering Quality
- **Production-Quality First**: Do not write "throwaway" code. From the first implementation, treat the codebase as a production artifact.
- **Strong Typing**: Leverage Python's type system pervasively to catch errors at static analysis time.
- **Research Correctness**: Never take engineering shortcuts that compromise the integrity or validity of the core ML experiments.
- **Maintainability over Premature Optimization**: Do not optimize for performance or latency until profiling or concrete evidence demonstrates a bottleneck.
- **Minimize Technical Debt**: Refactor continuously. If a workaround is necessary, it must be documented and scoped.
- **Strict Governance**: Every architectural deviation requires explicit architectural approval and must be documented in `DECISION_LOG.md`.
