"""Base interfaces and protocols for Verification Failure Analysis (M3.1)."""

from typing import Any, Protocol, runtime_checkable

from src.core.failure.failure_models import (
    FailureAnalysisDefinition,
    FailureAnalysisResult,
)


@runtime_checkable
class BaseFailureAnalyzer(Protocol):
    """Protocol for stateless execution of verification failure analysis."""

    def validate_compatibility(self, definition: FailureAnalysisDefinition) -> None:
        """Statically verifies compatibility of configuration settings."""
        ...

    def analyze(
        self,
        claim: str,
        verification_result: Any,
        definition: FailureAnalysisDefinition,
    ) -> FailureAnalysisResult:
        """Performs failure detection, classification, and diagnosis on verification artifacts."""
        ...
