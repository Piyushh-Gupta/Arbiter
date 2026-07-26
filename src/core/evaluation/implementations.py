"""Concrete implementations of evaluation engines."""

from src.core.evaluation.evaluation_models import (
    EvaluationDefinition,
    EvaluationMetadata,
    EvaluationMetric,
    EvaluationResult,
    RuleBasedEvaluationDefinition,
)
from src.core.exceptions import EvaluationConfigurationError, EvaluationExecutionError
from src.core.explainability.explainability_models import ExplanationResult
from src.core.failure_analysis.failure_analysis_models import FailureSeverity


class RuleBasedEvaluator:
    """Concrete engine that evaluates structural pipeline quality deterministically."""

    def validate_compatibility(self, definition: EvaluationDefinition) -> None:
        """Validates that the provided definition is a RuleBasedEvaluationDefinition."""
        if not isinstance(definition, RuleBasedEvaluationDefinition):
            raise EvaluationConfigurationError(
                f"RuleBasedEvaluator requires RuleBasedEvaluationDefinition, got {type(definition).__name__}"
            )

    def evaluate(
        self,
        explanation_result: ExplanationResult,
        definition: EvaluationDefinition,
    ) -> EvaluationResult:
        """
        Deterministically evaluates pipeline structural quality based on the ExplanationResult.
        """
        try:
            self.validate_compatibility(definition)

            metrics = []
            dr = explanation_result.decision_result
            ur = dr.uncertainty_result
            fa = ur.failure_analysis_result

            # 1. structural_completeness
            sev = fa.severity
            if sev == FailureSeverity.NONE:
                comp_score = 1.0
            elif sev in (FailureSeverity.LOW, FailureSeverity.MEDIUM):
                comp_score = 0.5
            else:
                comp_score = 0.0

            flags_str = (
                ", ".join(sorted(f.code for f in fa.failure_flags))
                if fa.failure_flags
                else "None"
            )
            metrics.append(
                EvaluationMetric(
                    identifier="structural_completeness",
                    title="Structural Completeness",
                    score=comp_score,
                    details=f"Failure Severity: {sev.value}. Flags: {flags_str}",
                )
            )

            # 2. uncertainty_confidence
            unc_conf = max(0.0, 1.0 - ur.score)
            metrics.append(
                EvaluationMetric(
                    identifier="uncertainty_confidence",
                    title="Uncertainty Confidence",
                    score=unc_conf,
                    details=f"Derived from uncertainty score {ur.score:.4f} (Level: {ur.level.value}).",
                )
            )

            # 3. explanation_richness (Information Coverage via structural identifiers)
            # Evaluator checks if core structural domains are represented in the explanation section identifiers.
            expected_domains = {"decision", "uncertainty", "failure", "verification"}
            covered_domains = set()
            for section in explanation_result.sections:
                identifier_lower = section.identifier.lower()
                for domain in expected_domains:
                    if domain in identifier_lower:
                        covered_domains.add(domain)

            rich_score = len(covered_domains) / len(expected_domains)
            metrics.append(
                EvaluationMetric(
                    identifier="explanation_richness",
                    title="Explanation Richness",
                    score=rich_score,
                    details=f"Covered structural domains: {len(covered_domains)}/{len(expected_domains)}.",
                )
            )

            return EvaluationResult(
                metrics=tuple(metrics),
                explanation_result=explanation_result,
                metadata=EvaluationMetadata(strategy_id="rule_based_evaluator"),
            )
        except Exception as e:
            if isinstance(e, EvaluationConfigurationError):
                raise
            raise EvaluationExecutionError(
                f"RuleBasedEvaluator execution failed: {e}"
            ) from e
