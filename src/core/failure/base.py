"""Base interfaces and protocols for Verification Failure Analysis (M3.3)."""

from typing import Any, Protocol, Sequence, runtime_checkable

from src.core.failure.failure_models import (
    AnalyzerExecutionResult,
    FailureAnalysisDefinition,
    FailureAnalysisInput,
    FailureAnalysisResult,
)


@runtime_checkable
class BaseFailureAnalyzer(Protocol):
    """Protocol for stateless execution of verification failure analysis."""

    @property
    def supported_categories(self) -> tuple[Any, ...]:
        """Returns the failure categories supported by this analyzer."""
        ...

    @property
    def runtime_metadata(self) -> Any:
        """Returns runtime execution provenance for reproducible diagnostics."""
        ...

    def validate_compatibility(self, definition: FailureAnalysisDefinition) -> None:
        """Statically verifies compatibility of configuration settings."""
        ...

    def analyze(
        self,
        input_data: FailureAnalysisInput,
    ) -> FailureAnalysisResult:
        """Performs failure detection, classification, and diagnosis on verification artifacts."""
        ...


@runtime_checkable
class FailureAggregationStrategy(Protocol):
    """Protocol for combining sequence of AnalyzerExecutionResult objects into a FailureAnalysisResult."""

    def aggregate(
        self,
        results: Sequence[AnalyzerExecutionResult],
        input_data: FailureAnalysisInput,
    ) -> FailureAnalysisResult:
        """Aggregates specialized analyzer execution outputs and builds the final diagnostic verdict."""
        ...
