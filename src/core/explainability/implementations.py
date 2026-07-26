"""Concrete implementations of explanation engines."""

from src.core.decision.decision_models import DecisionResult
from src.core.exceptions import ExplanationConfigurationError, ExplanationExecutionError
from src.core.explainability.explainability_models import (
    ExplanationDefinition,
    ExplanationMetadata,
    ExplanationResult,
    ExplanationSection,
    RuleBasedExplanationDefinition,
)


class RuleBasedExplainer:
    """Concrete engine that generates deterministic, rule-based explanations."""

    def validate_compatibility(self, definition: ExplanationDefinition) -> None:
        """Validates that the provided definition is a RuleBasedExplanationDefinition."""
        if not isinstance(definition, RuleBasedExplanationDefinition):
            raise ExplanationConfigurationError(
                f"RuleBasedExplainer requires RuleBasedExplanationDefinition, got {type(definition).__name__}"
            )

    def explain(
        self,
        claim: str,
        decision_result: DecisionResult,
        definition: ExplanationDefinition,
    ) -> ExplanationResult:
        """
        Deterministically generates an explanation based on the decision result's underlying state.
        """
        try:
            self.validate_compatibility(definition)

            sections = []

            # 1. Decision Rationale (Verbatim)
            sections.append(
                ExplanationSection(
                    identifier="decision_rationale",
                    title="Decision Rationale",
                    content=f"Action taken: {decision_result.action.value}. {decision_result.rationale}",
                )
            )

            # 2. Uncertainty Analysis
            ur = decision_result.uncertainty_result
            sections.append(
                ExplanationSection(
                    identifier="uncertainty_analysis",
                    title="Uncertainty Analysis",
                    content=f"Final Uncertainty Score: {ur.score:.4f} (Level: {ur.level.value}).",
                )
            )

            # 3. Uncertainty Factors (Conditional)
            if ur.factors:
                factors_str = ", ".join(
                    f.code for f in sorted(ur.factors, key=lambda x: x.code)
                )
                sections.append(
                    ExplanationSection(
                        identifier="uncertainty_factors",
                        title="Uncertainty Factors",
                        content=f"Active factors: {factors_str}",
                    )
                )

            # 4. Failure Analysis
            fa = ur.failure_analysis_result
            sections.append(
                ExplanationSection(
                    identifier="failure_analysis",
                    title="Failure Analysis",
                    content=f"Maximum Severity: {fa.severity.value}.",
                )
            )

            # 5. Failure Flags (Conditional)
            if fa.failure_flags:
                flags_str = ", ".join(
                    f.code for f in sorted(fa.failure_flags, key=lambda x: x.code)
                )
                sections.append(
                    ExplanationSection(
                        identifier="failure_flags",
                        title="Failure Flags",
                        content=f"Detected flags: {flags_str}",
                    )
                )

            # 6. Verification Result
            vr = fa.verification_result
            conf_str = f"{vr.confidence:.4f}" if vr.confidence is not None else "N/A"
            sections.append(
                ExplanationSection(
                    identifier="verification_result",
                    title="Verification Result",
                    content=f"Label: {vr.label.value} (Confidence: {conf_str}).",
                )
            )

            return ExplanationResult(
                sections=tuple(sections),
                decision_result=decision_result,
                metadata=ExplanationMetadata(strategy_id="rule_based_explainer"),
            )
        except Exception as e:
            if isinstance(e, ExplanationConfigurationError):
                raise
            raise ExplanationExecutionError(
                f"RuleBasedExplainer execution failed: {e}"
            ) from e
