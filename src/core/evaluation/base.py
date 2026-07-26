"""Core protocol defining evaluation engines."""

from typing import Protocol

from src.core.evaluation.evaluation_models import EvaluationDefinition, EvaluationResult
from src.core.explainability.explainability_models import ExplanationResult


class BaseEvaluator(Protocol):
    """
    Protocol for all evaluation engines.
    """

    def validate_compatibility(self, definition: EvaluationDefinition) -> None:
        """
        Validates whether the provided configuration is compatible with this evaluator.

        Args:
            definition: The configuration parameters to validate.

        Raises:
            EvaluationConfigurationError: If the configuration is incompatible.
        """
        ...

    def evaluate(
        self,
        explanation_result: ExplanationResult,
        definition: EvaluationDefinition,
    ) -> EvaluationResult:
        """
        Executes the evaluation strategy.

        Args:
            explanation_result: The complete pipeline execution state up to the explanation.
            definition: The pre-validated configuration parameters.

        Returns:
            EvaluationResult: The resulting evaluation metrics.

        Raises:
            EvaluationExecutionError: If execution fails unexpectedly.
        """
        ...
