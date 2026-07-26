"""Base protocol for uncertainty estimation strategies."""

from typing import Protocol, runtime_checkable

from src.core.failure_analysis.failure_analysis_models import FailureAnalysisResult
from src.core.uncertainty.uncertainty_models import (
    UncertaintyDefinition,
    UncertaintyResult,
)


@runtime_checkable
class BaseUncertaintyEstimator(Protocol):
    """Protocol for stateless execution of uncertainty estimation logic."""

    def validate_compatibility(self, definition: UncertaintyDefinition) -> None:
        """
        Statically verifies if this estimator supports the given definition.

        Args:
            definition: The configuration to validate.

        Raises:
            UncertaintyConfigurationError: If the definition is incompatible or malformed.
        """
        ...

    def estimate(
        self,
        claim: str,
        failure_analysis_result: FailureAnalysisResult,
        definition: UncertaintyDefinition,
    ) -> UncertaintyResult:
        """
        Executes uncertainty estimation logic.

        Args:
            claim: The normalized textual assertion.
            failure_analysis_result: The preceding immutable pipeline state.
            definition: The validated, immutable configuration parameters.

        Returns:
            UncertaintyResult: A fully materialized, immutable uncertainty record.

        Raises:
            UncertaintyExecutionError: If a runtime failure occurs during estimation.
        """
        ...
