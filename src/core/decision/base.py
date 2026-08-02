"""Base protocols for Decision Engine Architecture Modernization (M4.1)."""

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.core.decision.decision_models import DecisionResult


@runtime_checkable
class BaseDecisionEngine(Protocol):
    """Legacy protocol for decision engines for backward compatibility."""

    def validate_compatibility(self, definition: Any) -> None:
        """Validates configuration parameters."""
        ...

    def decide(
        self,
        claim: str,
        uncertainty_result: Any,
        definition: Any,
    ) -> Any:
        """Executes decision policy."""
        ...


@runtime_checkable
class BaseDecisionStrategy(Protocol):
    """Protocol for stateless execution of decision strategies."""

    def validate_compatibility(self, definition: Any) -> None:
        """Statically verifies compatibility of decision policy settings."""
        ...

    def decide(
        self,
        context_or_claim: Any,
        definition_or_unc: Any = None,
        definition: Any = None,
    ) -> "DecisionResult":
        """Executes decision policy evaluation over the provided decision context or parameters."""
        ...
