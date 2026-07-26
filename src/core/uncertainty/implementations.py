"""Concrete implementations of uncertainty estimators."""

from typing import cast

from src.core.exceptions import UncertaintyConfigurationError
from src.core.failure_analysis.failure_analysis_models import FailureAnalysisResult
from src.core.uncertainty.uncertainty_models import (
    ConfidenceUncertaintyDefinition,
    UncertaintyDefinition,
    UncertaintyFactor,
    UncertaintyLevel,
    UncertaintyMetadata,
    UncertaintyResult,
)


class ConfidenceUncertaintyEstimator:
    """Estimates uncertainty via strictly linear inversion of the verification confidence."""

    def validate_compatibility(self, definition: UncertaintyDefinition) -> None:
        """
        Validates that the provided definition is a ConfidenceUncertaintyDefinition.

        Args:
            definition: The configuration to validate.

        Raises:
            UncertaintyConfigurationError: If the definition is incompatible.
        """
        if not isinstance(definition, ConfidenceUncertaintyDefinition):
            raise UncertaintyConfigurationError(
                f"ConfidenceUncertaintyEstimator requires ConfidenceUncertaintyDefinition, got {type(definition).__name__}"
            )

    def estimate(
        self,
        claim: str,
        failure_analysis_result: FailureAnalysisResult,
        definition: UncertaintyDefinition,
    ) -> UncertaintyResult:
        """
        Executes confidence-based uncertainty estimation.

        Args:
            claim: The normalized textual assertion.
            failure_analysis_result: The preceding immutable pipeline state.
            definition: The validated, immutable configuration parameters.

        Returns:
            UncertaintyResult: The resulting uncertainty representation.
        """
        self.validate_compatibility(definition)
        conf_def = cast(ConfidenceUncertaintyDefinition, definition)

        confidence = failure_analysis_result.verification_result.confidence

        if confidence is None:
            score = 1.0
            factors = frozenset(
                [
                    UncertaintyFactor(
                        code="ABSENT_CONFIDENCE",
                        description="Verification confidence was missing, indicating absolute uncertainty.",
                    )
                ]
            )
        else:
            # Round to 6 decimal places to prevent IEEE 754 floating-point errors
            # (e.g., 1.0 - 0.7 = 0.30000000000000004) from violating strict boundaries.
            score = round(1.0 - confidence, 6)
            factors = frozenset()

        if score <= conf_def.none_threshold:
            level = UncertaintyLevel.NONE
        elif score <= conf_def.low_threshold:
            level = UncertaintyLevel.LOW
        elif score <= conf_def.medium_threshold:
            level = UncertaintyLevel.MEDIUM
        elif score <= conf_def.high_threshold:
            level = UncertaintyLevel.HIGH
        else:
            level = UncertaintyLevel.EXTREME

        return UncertaintyResult(
            level=level,
            score=score,
            factors=factors,
            failure_analysis_result=failure_analysis_result,
            metadata=UncertaintyMetadata(
                strategy_id="confidence_uncertainty_estimator"
            ),
        )
