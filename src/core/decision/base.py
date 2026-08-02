"""Base protocols for Decision Engine Architecture Modernization (M4.1 & M4.2)."""

from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from src.core.decision.decision_models import (
        DecisionExecutionContext,
        DecisionInput,
        DecisionPolicyGroup,
        DecisionResult,
    )


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
class BaseDecisionPolicyEngine(Protocol):
    """Protocol for stateless evaluation of decision policy groups and rules."""

    def validate_compatibility(self, definition: Any) -> None:
        """Validates that the provided definition is compatible with this policy engine."""
        ...

    def evaluate(
        self,
        input_data: "DecisionInput",
        policy_groups: Sequence["DecisionPolicyGroup"] | None = None,
    ) -> "DecisionExecutionContext":
        """Evaluates policy groups statelessly and returns immutable DecisionExecutionContext."""
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
