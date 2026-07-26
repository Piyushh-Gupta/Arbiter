"""Core protocol defining decision engines."""

from typing import Protocol

from src.core.decision.decision_models import DecisionDefinition, DecisionResult
from src.core.uncertainty.uncertainty_models import UncertaintyResult


class BaseDecisionEngine(Protocol):
    """
    Protocol for all deterministic decision engines.
    """

    def validate_compatibility(self, definition: DecisionDefinition) -> None:
        """
        Validates whether the provided configuration is compatible with this engine.

        Args:
            definition: The configuration parameters to validate.

        Raises:
            DecisionConfigurationError: If the configuration is incompatible.
        """
        ...

    def decide(
        self,
        claim: str,
        uncertainty_result: UncertaintyResult,
        definition: DecisionDefinition,
    ) -> DecisionResult:
        """
        Executes the deterministic decision policy.

        Args:
            claim: The normalized textual claim being evaluated.
            uncertainty_result: The preceding immutable pipeline state.
            definition: The pre-validated configuration parameters.

        Returns:
            DecisionResult: The resulting decision and rationale.

        Raises:
            DecisionExecutionError: If execution fails unexpectedly.
        """
        ...
