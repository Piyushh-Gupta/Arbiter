"""Base protocols for Decision Explainability & Audit Reporting (M4.6)."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BaseDecisionExplanationStrategy(Protocol):
    """Stateless protocol for post-decision explainability strategies."""

    @property
    def strategy_id(self) -> str:
        """Unique identifier for the explanation strategy."""
        ...

    def validate_compatibility(self, definition: Any) -> None:
        """Validates compatibility with the explanation definition."""
        ...

    def generate_explanation(self, context: Any, definition: Any) -> Any:
        """Generates a structured DecisionExplanation from the execution context."""
        ...
