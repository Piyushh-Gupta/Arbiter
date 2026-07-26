"""Concrete implementations of decision engines."""

from src.core.decision.decision_models import (
    DecisionAction,
    DecisionDefinition,
    DecisionMetadata,
    DecisionResult,
    ThresholdDecisionDefinition,
)
from src.core.exceptions import DecisionConfigurationError, DecisionExecutionError
from src.core.uncertainty.uncertainty_models import UncertaintyResult
from src.core.verification.verification_models import VerificationLabel


class ThresholdDecisionEngine:
    """Concrete engine that routes claims based on verification labels and continuous uncertainty thresholds."""

    def validate_compatibility(self, definition: DecisionDefinition) -> None:
        """
        Validates that the provided definition is a ThresholdDecisionDefinition.
        """
        if not isinstance(definition, ThresholdDecisionDefinition):
            raise DecisionConfigurationError(
                f"ThresholdDecisionEngine requires ThresholdDecisionDefinition, got {type(definition).__name__}"
            )

    def decide(
        self,
        claim: str,
        uncertainty_result: UncertaintyResult,
        definition: DecisionDefinition,
    ) -> DecisionResult:
        """
        Deterministically routes a claim based on verification label and uncertainty score.
        """
        try:
            self.validate_compatibility(definition)

            # Inform type checker of the guaranteed type
            assert isinstance(definition, ThresholdDecisionDefinition)

            score = uncertainty_result.score
            label = uncertainty_result.failure_analysis_result.verification_result.label

            if (
                label == VerificationLabel.SUPPORTS
                and score <= definition.accept_max_uncertainty
            ):
                action = DecisionAction.ACCEPT
                rationale = f"Claim supported with uncertainty ({score}) <= threshold ({definition.accept_max_uncertainty})."

            elif (
                label == VerificationLabel.REFUTES
                and score <= definition.reject_max_uncertainty
            ):
                action = DecisionAction.REJECT
                rationale = f"Claim refuted with uncertainty ({score}) <= threshold ({definition.reject_max_uncertainty})."

            else:
                action = DecisionAction.ESCALATE
                rationale = f"Uncertainty ({score}) exceeds acceptable thresholds or label ({label.value}) lacks deterministic routing."

            return DecisionResult(
                action=action,
                rationale=rationale,
                uncertainty_result=uncertainty_result,
                metadata=DecisionMetadata(strategy_id="threshold_decision_engine"),
            )
        except Exception as e:
            if isinstance(e, DecisionConfigurationError):
                raise
            raise DecisionExecutionError(
                f"ThresholdDecisionEngine execution failed: {e}"
            ) from e
