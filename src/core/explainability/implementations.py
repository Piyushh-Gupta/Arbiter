"""Concrete implementations of explanation engines."""

from typing import Any

from src.core.decision.decision_models import DecisionResult
from src.core.exceptions import ExplanationConfigurationError, ExplanationExecutionError
from src.core.explainability.explainability_models import (
    ExplanationDefinition,
    ExplanationMetadata,
    ExplanationResult,
    ExplanationSection,
    RuleBasedExplanationDefinition,
)
from src.core.explainability.explanation_models import (
    ContributionAnalysis,
    DecisionTrace,
    EvidenceAttribution,
    ExplanationTrace,
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
            act_val = (
                decision_result.action.value
                if hasattr(decision_result.action, "value")
                else str(decision_result.action)
            )
            sections.append(
                ExplanationSection(
                    identifier="decision_rationale",
                    title="Decision Rationale",
                    content=f"Action taken: {act_val}. {decision_result.rationale}",
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


class EvidenceAttributionStrategy:
    """Explains supporting, contradicting, and ignored passages."""

    def validate_compatibility(self, definition: ExplanationDefinition) -> None:
        from src.core.explainability.explanation_models import (
            VerificationExplanationDefinition,
        )

        if not isinstance(definition, VerificationExplanationDefinition):
            raise ExplanationConfigurationError(
                f"EvidenceAttributionStrategy requires VerificationExplanationDefinition, got {type(definition).__name__}"
            )

    def explain(
        self,
        verification_result: Any,
        calibration_result: Any,
        evidence_bundle: Any,
        aggregation_trace: Any,
        definition: ExplanationDefinition,
    ) -> ExplanationResult:
        try:
            self.validate_compatibility(definition)

            supporting = set(verification_result.supporting_passages or ())
            contradicting = set(verification_result.contradicting_passages or ())
            all_spans = [p.span_id for p in evidence_bundle.passages]
            ignored = tuple(
                sid
                for sid in all_spans
                if sid not in supporting and sid not in contradicting
            )

            contrib_weights = aggregation_trace.weighting_decisions or {}

            attribution = EvidenceAttribution(
                supporting_passages=verification_result.supporting_passages or (),
                contradicting_passages=verification_result.contradicting_passages or (),
                ignored_passages=ignored,
                contribution_weights=contrib_weights,
            )

            lines = [
                f"Supporting Passages: {', '.join(attribution.supporting_passages) or 'None'}",
                f"Contradicting Passages: {', '.join(attribution.contradicting_passages) or 'None'}",
                f"Ignored Passages: {', '.join(attribution.ignored_passages) or 'None'}",
            ]
            content = "\n".join(lines)

            section = ExplanationSection(
                identifier="evidence_attribution",
                title="Evidence Attribution Summary",
                content=content,
            )

            return ExplanationResult(
                sections=(section,),
                metadata=ExplanationMetadata(strategy_id="evidence_attribution"),
                verification_result=verification_result,
                calibration_result=calibration_result,
                evidence_attribution=attribution,
            )
        except Exception as e:
            if isinstance(e, ExplanationConfigurationError):
                raise
            raise ExplanationExecutionError(
                f"EvidenceAttributionStrategy execution failed: {e}"
            ) from e


class DecisionTraceStrategy:
    """Explains decision trace, aggregation strategy, calibration, and contradiction resolution."""

    def validate_compatibility(self, definition: ExplanationDefinition) -> None:
        from src.core.explainability.explanation_models import (
            VerificationExplanationDefinition,
        )

        if not isinstance(definition, VerificationExplanationDefinition):
            raise ExplanationConfigurationError(
                f"DecisionTraceStrategy requires VerificationExplanationDefinition, got {type(definition).__name__}"
            )

    def explain(
        self,
        verification_result: Any,
        calibration_result: Any,
        evidence_bundle: Any,
        aggregation_trace: Any,
        definition: ExplanationDefinition,
    ) -> ExplanationResult:
        try:
            self.validate_compatibility(definition)

            raw_conf = 0.0
            if verification_result.verified_passages:
                confs = []
                for vp in verification_result.verified_passages:
                    c = getattr(vp, "confidence", None)
                    if c is None:
                        c = max(
                            getattr(vp, "supports_score", 0.0),
                            getattr(vp, "refutes_score", 0.0),
                            getattr(vp, "not_enough_info_score", 0.0),
                        )
                    confs.append(c)
                raw_conf = sum(confs) / len(confs) if confs else 0.0
            else:
                raw_conf = calibration_result.original_confidence or 0.0

            evolution = (
                raw_conf,
                calibration_result.original_confidence or 0.0,
                calibration_result.calibrated_confidence,
            )

            conflict_res = "No contradiction active."
            if verification_result.conflict_analysis:
                conflict_res = (
                    verification_result.conflict_analysis.resolution_rationale
                )

            trace = DecisionTrace(
                aggregation_strategy=aggregation_trace.aggregation_strategy,
                calibration_strategy=str(
                    calibration_result.calibration_trace.applied_strategy.value
                ),
                confidence_evolution=evolution,
                contradiction_resolution=conflict_res,
            )

            content = (
                f"Aggregation Strategy: {trace.aggregation_strategy}\n"
                f"Calibration Strategy: {trace.calibration_strategy}\n"
                f"Contradiction Resolution: {trace.contradiction_resolution}"
            )

            section = ExplanationSection(
                identifier="decision_trace",
                title="Decision Trace Summary",
                content=content,
            )

            return ExplanationResult(
                sections=(section,),
                metadata=ExplanationMetadata(strategy_id="decision_trace"),
                verification_result=verification_result,
                calibration_result=calibration_result,
                decision_trace=trace,
            )
        except Exception as e:
            if isinstance(e, ExplanationConfigurationError):
                raise
            raise ExplanationExecutionError(
                f"DecisionTraceStrategy execution failed: {e}"
            ) from e


class ConfidenceExplanationStrategy:
    """Explains confidence adjustments and uncertainty levels."""

    def validate_compatibility(self, definition: ExplanationDefinition) -> None:
        from src.core.explainability.explanation_models import (
            VerificationExplanationDefinition,
        )

        if not isinstance(definition, VerificationExplanationDefinition):
            raise ExplanationConfigurationError(
                f"ConfidenceExplanationStrategy requires VerificationExplanationDefinition, got {type(definition).__name__}"
            )

    def explain(
        self,
        verification_result: Any,
        calibration_result: Any,
        evidence_bundle: Any,
        aggregation_trace: Any,
        definition: ExplanationDefinition,
    ) -> ExplanationResult:
        try:
            self.validate_compatibility(definition)

            orig = calibration_result.original_confidence or 0.0
            cal = calibration_result.calibrated_confidence
            unc = calibration_result.uncertainty_estimate

            content = (
                f"Original Aggregated Confidence: {orig:.4f}\n"
                f"Calibrated Confidence: {cal:.4f}\n"
                f"Uncertainty Estimate: {unc:.4f}"
            )

            section = ExplanationSection(
                identifier="confidence_explanation",
                title="Confidence & Uncertainty Analysis",
                content=content,
            )

            return ExplanationResult(
                sections=(section,),
                metadata=ExplanationMetadata(strategy_id="confidence_explanation"),
                verification_result=verification_result,
                calibration_result=calibration_result,
            )
        except Exception as e:
            if isinstance(e, ExplanationConfigurationError):
                raise
            raise ExplanationExecutionError(
                f"ConfidenceExplanationStrategy execution failed: {e}"
            ) from e


class CompositeExplanationStrategy:
    """Composes EvidenceAttribution, DecisionTrace, and ConfidenceExplanation strategies."""

    def __init__(self) -> None:
        self.attribution_strategy = EvidenceAttributionStrategy()
        self.trace_strategy = DecisionTraceStrategy()
        self.confidence_strategy = ConfidenceExplanationStrategy()

    def validate_compatibility(self, definition: ExplanationDefinition) -> None:
        from src.core.explainability.explanation_models import (
            VerificationExplanationDefinition,
        )

        if not isinstance(definition, VerificationExplanationDefinition):
            raise ExplanationConfigurationError(
                f"CompositeExplanationStrategy requires VerificationExplanationDefinition, got {type(definition).__name__}"
            )

    def explain(
        self,
        verification_result: Any,
        calibration_result: Any,
        evidence_bundle: Any,
        aggregation_trace: Any,
        definition: ExplanationDefinition,
    ) -> ExplanationResult:
        try:
            self.validate_compatibility(definition)

            # 1. Delegate explanation generations
            attr_res = self.attribution_strategy.explain(
                verification_result,
                calibration_result,
                evidence_bundle,
                aggregation_trace,
                definition,
            )
            trace_res = self.trace_strategy.explain(
                verification_result,
                calibration_result,
                evidence_bundle,
                aggregation_trace,
                definition,
            )
            conf_res = self.confidence_strategy.explain(
                verification_result,
                calibration_result,
                evidence_bundle,
                aggregation_trace,
                definition,
            )

            # 2. Combine sections
            sections = attr_res.sections + trace_res.sections + conf_res.sections

            # 3. Numeric Contribution Analysis
            passages = evidence_bundle.passages
            ret_contrib = (
                sum(p.score for p in passages) / len(passages) if passages else 0.0
            )

            verified_passages = verification_result.verified_passages
            ver_contrib = 0.0
            if verified_passages:
                confs = []
                for vp in verified_passages:
                    c = getattr(vp, "confidence", None)
                    if c is None:
                        c = max(
                            getattr(vp, "supports_score", 0.0),
                            getattr(vp, "refutes_score", 0.0),
                            getattr(vp, "not_enough_info_score", 0.0),
                        )
                    confs.append(c)
                ver_contrib = sum(confs) / len(confs) if confs else 0.0

            agg_contrib = (
                sum(aggregation_trace.weighting_decisions.values())
                if aggregation_trace.weighting_decisions
                else 0.0
            )

            cal_diff = abs(
                (calibration_result.original_confidence or 0.0)
                - calibration_result.calibrated_confidence
            )

            contribution = ContributionAnalysis(
                retrieval_contribution=ret_contrib,
                verification_contribution=ver_contrib,
                aggregation_contribution=agg_contrib,
                calibration_contribution=cal_diff,
            )

            # 4. Explanation Trace
            v_profile = (
                verification_result.model_metadata.model_identifier
                if hasattr(verification_result, "model_metadata")
                and verification_result.model_metadata
                else "default"
            )
            c_profile = str(calibration_result.calibration_trace.applied_strategy.value)

            trace = ExplanationTrace(
                explanation_strategy="COMPOSITE",
                verification_profile=v_profile,
                aggregation_profile=aggregation_trace.aggregation_strategy,
                calibration_profile=c_profile,
                evidence_traversal=aggregation_trace.ordered_evaluation_sequence,
                execution_order=("Attribution", "Trace", "Confidence"),
            )

            return ExplanationResult(
                sections=sections,
                metadata=ExplanationMetadata(strategy_id="composite_explanation"),
                verification_result=verification_result,
                calibration_result=calibration_result,
                evidence_attribution=attr_res.evidence_attribution,
                decision_trace=trace_res.decision_trace,
                contribution_analysis=contribution,
                explanation_trace=trace,
            )
        except Exception as e:
            if isinstance(e, ExplanationConfigurationError):
                raise
            raise ExplanationExecutionError(
                f"CompositeExplanationStrategy execution failed: {e}"
            ) from e
