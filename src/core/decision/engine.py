"""Pure delegating orchestrator for decision engines."""

from typing import Any, cast

from src.core.decision.decision_models import DecisionResult
from src.core.exceptions import DecisionExecutionError


class DecisionEngine:
    """
    Stateless orchestrator that delegates execution to concrete decision engines.
    """

    def decide(
        self,
        claim: Any,
        uncertainty_result: Any = None,
        definition: Any = None,
        strategy: Any = None,
    ) -> DecisionResult:
        """
        Delegates the decision execution to the provided strategy or engine.
        """
        executable = strategy or definition
        if hasattr(executable, "decide"):
            res = executable.decide(claim, uncertainty_result, definition)
            return cast(DecisionResult, res)
        raise DecisionExecutionError("Invalid strategy provided to DecisionEngine.")
