"""Pure delegating orchestrator for evaluation engines."""

from src.core.evaluation.base import BaseEvaluator
from src.core.evaluation.evaluation_models import EvaluationDefinition, EvaluationResult
from src.core.explainability.explainability_models import ExplanationResult


class Evaluator:
    """
    Stateless orchestrator that delegates execution to concrete evaluators.
    """

    def evaluate(
        self,
        explanation_result: ExplanationResult,
        definition: EvaluationDefinition,
        strategy: BaseEvaluator,
    ) -> EvaluationResult:
        """
        Delegates evaluation generation to the provided strategy.

        Note: Compatibility validation is deliberately omitted from this orchestrator,
        leaving it exactly to profile construction, preserving orchestrator purity.

        Args:
            explanation_result: The accumulated pipeline state and explanation.
            definition: The assumed-valid configuration parameters.
            strategy: The concrete, stateless strategy.

        Returns:
            EvaluationResult: The final evaluation result.
        """
        return strategy.evaluate(explanation_result, definition)
