"""Core protocol defining explanation engines."""

from typing import Protocol

from src.core.decision.decision_models import DecisionResult
from src.core.explainability.explainability_models import (
    ExplanationDefinition,
    ExplanationResult,
)


class BaseExplainer(Protocol):
    """
    Protocol for all deterministic explanation engines.
    """

    def validate_compatibility(self, definition: ExplanationDefinition) -> None:
        """
        Validates whether the provided configuration is compatible with this explainer.

        Args:
            definition: The configuration parameters to validate.

        Raises:
            ExplanationConfigurationError: If the configuration is incompatible.
        """
        ...

    def explain(
        self,
        claim: str,
        decision_result: DecisionResult,
        definition: ExplanationDefinition,
    ) -> ExplanationResult:
        """
        Executes the explanation generation policy.

        Args:
            claim: The normalized textual claim being explained.
            decision_result: The preceding immutable pipeline state and final routing decision.
            definition: The pre-validated configuration parameters.

        Returns:
            ExplanationResult: The resulting explanation sections.

        Raises:
            ExplanationExecutionError: If execution fails unexpectedly.
        """
        ...
