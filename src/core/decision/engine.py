"""Pure delegating orchestrator for decision engines."""

from src.core.decision.base import BaseDecisionEngine
from src.core.decision.decision_models import DecisionDefinition, DecisionResult
from src.core.uncertainty.uncertainty_models import UncertaintyResult


class DecisionEngine:
    """
    Stateless orchestrator that delegates execution to concrete decision engines.
    """

    def decide(
        self,
        claim: str,
        uncertainty_result: UncertaintyResult,
        definition: DecisionDefinition,
        strategy: BaseDecisionEngine,
    ) -> DecisionResult:
        """
        Delegates the decision execution to the provided strategy.

        Note: Compatibility validation is deliberately omitted from this orchestrator,
        leaving it exactly to profile construction, preserving orchestrator purity.

        Args:
            claim: The normalized textual assertion.
            uncertainty_result: The accumulated pipeline state.
            definition: The assumed-valid configuration parameters.
            strategy: The concrete, stateless strategy.

        Returns:
            DecisionResult: The final routing decision.
        """
        return strategy.decide(claim, uncertainty_result, definition)
