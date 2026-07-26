"""Pure delegating orchestrator for explanation engines."""

from src.core.decision.decision_models import DecisionResult
from src.core.explainability.base import BaseExplainer
from src.core.explainability.explainability_models import (
    ExplanationDefinition,
    ExplanationResult,
)


class Explainer:
    """
    Stateless orchestrator that delegates execution to concrete explainers.
    """

    def explain(
        self,
        claim: str,
        decision_result: DecisionResult,
        definition: ExplanationDefinition,
        strategy: BaseExplainer,
    ) -> ExplanationResult:
        """
        Delegates explanation generation to the provided strategy.

        Note: Compatibility validation is deliberately omitted from this orchestrator,
        leaving it exactly to profile construction, preserving orchestrator purity.

        Args:
            claim: The normalized textual assertion.
            decision_result: The accumulated pipeline state and routing decision.
            definition: The assumed-valid configuration parameters.
            strategy: The concrete, stateless strategy.

        Returns:
            ExplanationResult: The final explanation result.
        """
        return strategy.explain(claim, decision_result, definition)
