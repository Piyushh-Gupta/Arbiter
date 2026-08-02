"""Base protocols for Failure Explainability & Reporting subsystem (M3.7)."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from src.core.failure.failure_models import (
    FailureAnalysisResult,
    FailureCorrelationResult,
    RootCauseResult,
    SeverityEvaluationResult,
)

if TYPE_CHECKING:
    from src.core.failure.explainability.explanation_models import (
        FailureExplanationDefinition,
        FailureExplanationResult,
    )


@runtime_checkable
class BaseFailureExplanationStrategy(Protocol):
    """Protocol for stateless execution of failure explanation strategies."""

    def validate_compatibility(
        self, definition: "FailureExplanationDefinition"
    ) -> None:
        """Statically verifies compatibility of explanation configuration settings."""
        ...

    def explain(
        self,
        analysis_result: FailureAnalysisResult,
        correlation_result: FailureCorrelationResult | None = None,
        root_cause_result: RootCauseResult | None = None,
        severity_result: SeverityEvaluationResult | None = None,
        definition: "FailureExplanationDefinition | None" = None,
    ) -> "FailureExplanationResult":
        """Generates a structured failure explanation result from diagnostic inputs."""
        ...
