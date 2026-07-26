"""Base protocol for failure analysis strategies."""

from typing import Protocol, runtime_checkable

from src.core.verification.verification_models import VerificationResult

from .failure_analysis_models import FailureAnalysisDefinition, FailureAnalysisResult


@runtime_checkable
class BaseFailureAnalyzer(Protocol):
    """Protocol for stateless execution of failure analysis logic."""

    def validate_compatibility(self, definition: FailureAnalysisDefinition) -> None:
        """
        Statically verifies if this analyzer supports the given definition.

        Args:
            definition: The configuration to validate.

        Raises:
            FailureAnalysisConfigurationError: If the definition is incompatible or malformed.
        """
        ...

    def analyze(
        self,
        claim: str,
        verification_result: VerificationResult,
        definition: FailureAnalysisDefinition,
    ) -> FailureAnalysisResult:
        """
        Executes failure analysis logic.

        Args:
            claim: The normalized textual assertion.
            verification_result: The complete immutable verification pipeline output.
            definition: The validated, immutable configuration parameters.

        Returns:
            FailureAnalysisResult: A fully materialized, immutable diagnostic record.
        """
        ...
