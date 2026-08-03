"""Base protocols for Decision Engine Architecture Modernization (M4.1 & M4.2)."""

from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from src.core.decision.decision_models import (
        DecisionContext,
        DecisionDefinition,
        DecisionExecutionContext,
        DecisionInput,
        DecisionMetrics,
        DecisionPolicyGroup,
        DecisionResult,
        RiskEvaluation,
    )


@runtime_checkable
class BaseDecisionMetricPolicy(Protocol):
    """Protocol for pluggable decision metric evaluation policies."""

    @property
    def policy_id(self) -> str:
        """Unique identifier for the metric policy."""
        ...

    def validate_compatibility(self, definition: "DecisionDefinition") -> None:
        """Validates compatibility with the provided decision configuration."""
        ...

    def evaluate_metrics(
        self, context: "DecisionContext", definition: "DecisionDefinition"
    ) -> "DecisionMetrics":
        """Computes or resolves confidence and uncertainty metrics from context."""
        ...


@runtime_checkable
class BaseRiskPolicy(Protocol):
    """Protocol for pluggable decision risk policies."""

    @property
    def policy_id(self) -> str:
        """Unique identifier for the risk policy."""
        ...

    def validate_compatibility(self, definition: "DecisionDefinition") -> None:
        """Validates compatibility with the provided decision configuration."""
        ...

    def evaluate_risk(
        self,
        context: "DecisionContext",
        metrics: "DecisionMetrics",
        definition: "DecisionDefinition",
    ) -> "RiskEvaluation":
        """Evaluates operational risk score, adjusted confidence/uncertainty, and traces."""
        ...


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
