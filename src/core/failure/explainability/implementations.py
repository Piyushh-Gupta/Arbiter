"""Concrete explanation strategies for Failure Explainability & Reporting subsystem (M3.7)."""

from datetime import datetime, timezone
from typing import Sequence

from src.core.exceptions import FailureAnalysisConfigurationError
from src.core.failure.explainability.base import BaseFailureExplanationStrategy
from src.core.failure.explainability.explanation_models import (
    FailureDecisionTrace,
    FailureEvidenceExplanation,
    FailureExplanationDefinition,
    FailureExplanationMetadata,
    FailureExplanationResult,
    compute_explanation_fingerprint,
)
from src.core.failure.failure_models import (
    FailureAnalysisResult,
    FailureCorrelationResult,
    RootCauseResult,
    SeverityEvaluationResult,
)


class SummaryExplanationStrategy(BaseFailureExplanationStrategy):
    """Stateless strategy that generates concise structured operational summaries."""

    def validate_compatibility(self, definition: FailureExplanationDefinition) -> None:
        """Validates that the explanation definition is compatible."""
        if not isinstance(definition, FailureExplanationDefinition):
            raise FailureAnalysisConfigurationError(
                "Invalid definition type for SummaryExplanationStrategy."
            )

    def explain(
        self,
        analysis_result: FailureAnalysisResult,
        correlation_result: FailureCorrelationResult | None = None,
        root_cause_result: RootCauseResult | None = None,
        severity_result: SeverityEvaluationResult | None = None,
        definition: FailureExplanationDefinition | None = None,
    ) -> FailureExplanationResult:
        """Generates structured summary data from diagnostic inputs."""
        effective_def = definition or FailureExplanationDefinition()
        self.validate_compatibility(effective_def)

        category = analysis_result.classification.category.value
        diag_summary = analysis_result.diagnostic.diagnostic_summary
        summary_text = f"Failure [{category}]: {diag_summary}"

        details_lines: list[str] = []
        if effective_def.include_severity and severity_result:
            details_lines.append(
                f"Severity Level: {severity_result.overall_severity.value}"
            )
            if severity_result.escalation_required:
                details_lines.append(
                    f"Escalation Required: True ({severity_result.escalation_reason})"
                )

        if effective_def.include_root_cause and root_cause_result:
            details_lines.append(
                f"Primary Root Cause: {root_cause_result.primary_root_cause}"
            )
            if root_cause_result.contributing_failures:
                contrib_str = ", ".join(root_cause_result.contributing_failures)
                details_lines.append(f"Contributing Failures: {contrib_str}")

        if effective_def.include_correlation and correlation_result:
            details_lines.append(f"Correlation Summary: {correlation_result.summary}")

        detailed_explanation = "\n".join(details_lines)

        # Build evidence explanation
        supporting_diags = (analysis_result.diagnostic.diagnostic_summary,)
        contrib_failures = (
            root_cause_result.contributing_failures if root_cause_result else ()
        )
        evidence_trace = (
            f"Inspected {len(analysis_result.diagnostic.affected_artifacts)} artifacts.",
        )

        evidence = FailureEvidenceExplanation(
            supporting_diagnostics=supporting_diags,
            contributing_failures=contrib_failures,
            ignored_failures=(),
            evidence_trace=evidence_trace,
        )

        # Build decision trace
        corr_path = (
            tuple(c.correlation_id for c in correlation_result.correlation_graph)
            if correlation_result
            else ()
        )
        attr_path = root_cause_result.dependency_path if root_cause_result else ()
        sev_path = severity_result.policy_trace if severity_result else ()
        reasoning = (
            f"Classified failure as {category}.",
            f"Root cause evaluated as {root_cause_result.primary_root_cause if root_cause_result else 'UNKNOWN'}.",
        )

        decision_trace = FailureDecisionTrace(
            correlation_path=corr_path,
            attribution_path=attr_path,
            severity_policy_path=sev_path,
            reasoning_chain=reasoning,
        )

        metadata = FailureExplanationMetadata(
            strategy_id="summary_explanation_strategy",
            configuration_fingerprint=compute_explanation_fingerprint(effective_def),
            schema_version="1.0",
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        return FailureExplanationResult(
            summary=summary_text,
            detailed_explanation=detailed_explanation,
            evidence_explanation=evidence,
            decision_trace=decision_trace,
            metadata=metadata,
        )


class DecisionTraceExplanationStrategy(BaseFailureExplanationStrategy):
    """Stateless strategy that generates structured step-by-step reasoning chains."""

    def validate_compatibility(self, definition: FailureExplanationDefinition) -> None:
        """Validates that the explanation definition is compatible."""
        if not isinstance(definition, FailureExplanationDefinition):
            raise FailureAnalysisConfigurationError(
                "Invalid definition type for DecisionTraceExplanationStrategy."
            )

    def explain(
        self,
        analysis_result: FailureAnalysisResult,
        correlation_result: FailureCorrelationResult | None = None,
        root_cause_result: RootCauseResult | None = None,
        severity_result: SeverityEvaluationResult | None = None,
        definition: FailureExplanationDefinition | None = None,
    ) -> FailureExplanationResult:
        """Generates structured decision trace data from diagnostic inputs."""
        effective_def = definition or FailureExplanationDefinition()
        self.validate_compatibility(effective_def)

        summary_text = f"Decision Trace Explanation for {analysis_result.classification.category.value}"

        reasoning_steps: list[str] = [
            f"Step 1: Analyzed failure in subsystem {analysis_result.classification.affected_subsystem}.",
            f"Step 2: Diagnostic summary - {analysis_result.diagnostic.diagnostic_summary}.",
        ]

        corr_path: list[str] = []
        if correlation_result:
            for edge in correlation_result.correlation_graph:
                corr_path.append(f"{edge.source_failure} -> {edge.target_failure}")
            reasoning_steps.append(
                f"Step 3: Graph correlation identified {len(correlation_result.root_failures)} root failures."
            )

        attr_path: list[str] = []
        if root_cause_result:
            attr_path.extend(root_cause_result.dependency_path)
            reasoning_steps.append(
                f"Step 4: Root cause attributed to {root_cause_result.primary_root_cause} with confidence {root_cause_result.attribution_confidence}."
            )

        sev_path: list[str] = []
        if severity_result:
            sev_path.extend(severity_result.policy_trace)
            reasoning_steps.append(
                f"Step 5: Severity evaluated to {severity_result.overall_severity.value}."
            )

        evidence = FailureEvidenceExplanation(
            supporting_diagnostics=(analysis_result.diagnostic.diagnostic_summary,),
            contributing_failures=root_cause_result.contributing_failures
            if root_cause_result
            else (),
            ignored_failures=(),
            evidence_trace=tuple(analysis_result.trace.diagnostic_sequence),
        )

        decision_trace = FailureDecisionTrace(
            correlation_path=tuple(corr_path),
            attribution_path=tuple(attr_path),
            severity_policy_path=tuple(sev_path),
            reasoning_chain=tuple(reasoning_steps),
        )

        metadata = FailureExplanationMetadata(
            strategy_id="decision_trace_explanation_strategy",
            configuration_fingerprint=compute_explanation_fingerprint(effective_def),
            schema_version="1.0",
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        return FailureExplanationResult(
            summary=summary_text,
            detailed_explanation="\n".join(reasoning_steps),
            evidence_explanation=evidence,
            decision_trace=decision_trace,
            metadata=metadata,
        )


class CompositeFailureExplanationStrategy(BaseFailureExplanationStrategy):
    """Stateless strategy that delegates to child strategies and merges structured outputs only."""

    def __init__(self, strategies: Sequence[BaseFailureExplanationStrategy]) -> None:
        if not strategies:
            raise FailureAnalysisConfigurationError(
                "CompositeFailureExplanationStrategy requires at least one child strategy."
            )
        self.strategies = tuple(strategies)

    def validate_compatibility(self, definition: FailureExplanationDefinition) -> None:
        """Validates compatibility of all constituent strategies."""
        if not isinstance(definition, FailureExplanationDefinition):
            raise FailureAnalysisConfigurationError(
                "Invalid definition type for CompositeFailureExplanationStrategy."
            )
        for s in self.strategies:
            s.validate_compatibility(definition)

    def explain(
        self,
        analysis_result: FailureAnalysisResult,
        correlation_result: FailureCorrelationResult | None = None,
        root_cause_result: RootCauseResult | None = None,
        severity_result: SeverityEvaluationResult | None = None,
        definition: FailureExplanationDefinition | None = None,
    ) -> FailureExplanationResult:
        """Runs child strategies and merges structured explanation objects deterministically."""
        effective_def = definition or FailureExplanationDefinition()
        self.validate_compatibility(effective_def)

        child_results: list[FailureExplanationResult] = [
            s.explain(
                analysis_result,
                correlation_result,
                root_cause_result,
                severity_result,
                effective_def,
            )
            for s in self.strategies
        ]

        summary_text = child_results[0].summary
        detailed_parts = [
            r.detailed_explanation for r in child_results if r.detailed_explanation
        ]
        detailed_explanation = "\n\n".join(detailed_parts)

        # Helper to merge & deduplicate tuples deterministically while keeping insertion order
        def _merge_tuples(*tuples: tuple[str, ...]) -> tuple[str, ...]:
            seen: set[str] = set()
            result: list[str] = []
            for t in tuples:
                for item in t:
                    if item not in seen:
                        seen.add(item)
                        result.append(item)
            return tuple(result)

        merged_evidence = FailureEvidenceExplanation(
            supporting_diagnostics=_merge_tuples(
                *(r.evidence_explanation.supporting_diagnostics for r in child_results)
            ),
            contributing_failures=_merge_tuples(
                *(r.evidence_explanation.contributing_failures for r in child_results)
            ),
            ignored_failures=_merge_tuples(
                *(r.evidence_explanation.ignored_failures for r in child_results)
            ),
            evidence_trace=_merge_tuples(
                *(r.evidence_explanation.evidence_trace for r in child_results)
            ),
        )

        merged_decision_trace = FailureDecisionTrace(
            correlation_path=_merge_tuples(
                *(r.decision_trace.correlation_path for r in child_results)
            ),
            attribution_path=_merge_tuples(
                *(r.decision_trace.attribution_path for r in child_results)
            ),
            severity_policy_path=_merge_tuples(
                *(r.decision_trace.severity_policy_path for r in child_results)
            ),
            reasoning_chain=_merge_tuples(
                *(r.decision_trace.reasoning_chain for r in child_results)
            ),
        )

        metadata = FailureExplanationMetadata(
            strategy_id="composite_failure_explanation_strategy",
            configuration_fingerprint=compute_explanation_fingerprint(effective_def),
            schema_version="1.0",
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        return FailureExplanationResult(
            summary=summary_text,
            detailed_explanation=detailed_explanation,
            evidence_explanation=merged_evidence,
            decision_trace=merged_decision_trace,
            metadata=metadata,
        )
