"""Concrete verification failure analysis strategies (M3.1)."""

from typing import Any

from src.core.exceptions import FailureAnalysisConfigurationError
from src.core.failure.base import BaseFailureAnalyzer
from src.core.failure.failure_models import (
    FailureAnalysisDefinition,
    FailureAnalysisResult,
    FailureCategory,
    FailureClassification,
    FailureDiagnostic,
    FailureRootCause,
    FailureSeverity,
    FailureTrace,
)


class DefaultFailureAnalyzer(BaseFailureAnalyzer):
    """Composably executes failure checks across evidence retrieval and verification outcomes."""

    def validate_compatibility(self, definition: FailureAnalysisDefinition) -> None:
        if not isinstance(definition, FailureAnalysisDefinition):
            raise FailureAnalysisConfigurationError(
                "DefaultFailureAnalyzer requires FailureAnalysisDefinition."
            )

    def analyze(
        self,
        claim: str,
        verification_result: Any,
        definition: FailureAnalysisDefinition,
    ) -> FailureAnalysisResult:
        if not isinstance(definition, FailureAnalysisDefinition):
            raise FailureAnalysisConfigurationError(
                "DefaultFailureAnalyzer requires FailureAnalysisDefinition."
            )

        category = FailureCategory.UNKNOWN
        severity = FailureSeverity.INFO
        root_cause = FailureRootCause.UNKNOWN
        diag_summary = "Diagnosis completed with no issues found."

        # 1. Inspect Retrieval Artifacts
        bundle = getattr(verification_result, "evidence_bundle", None)
        passages = getattr(bundle, "passages", ()) if bundle is not None else ()

        if not passages:
            category = FailureCategory.RETRIEVAL
            severity = FailureSeverity.CRITICAL
            root_cause = FailureRootCause.MISSING_EVIDENCE
            diag_summary = "No evidence passages retrieved."
        else:
            # 2. Inspect Verification Artifacts
            confidence = getattr(verification_result, "confidence", None)
            if confidence is not None and confidence < 0.5:
                category = FailureCategory.VERIFICATION
                severity = FailureSeverity.HIGH
                root_cause = FailureRootCause.LOW_CONFIDENCE
                diag_summary = f"Low verification confidence: {confidence:.2f} < 0.5."
            else:
                # 3. Check for Contradictory Evidence
                contradicting = getattr(
                    verification_result, "contradicting_passages", ()
                )
                supporting = getattr(verification_result, "supporting_passages", ())
                if contradicting and supporting:
                    category = FailureCategory.AGGREGATION
                    severity = FailureSeverity.MEDIUM
                    root_cause = FailureRootCause.CONTRADICTORY_EVIDENCE
                    diag_summary = "Contradictory evidence detected in passages."

        classification = FailureClassification(
            category=category,
            severity=severity,
            affected_subsystem="verification",
        )

        diagnostic = FailureDiagnostic(
            root_cause=root_cause,
            diagnostic_summary=diag_summary,
            affected_artifacts=("evidence_bundle", "verification_result"),
            recovery_recommendation=(
                "Verify retrieval recall or verification thresholds"
                if severity != FailureSeverity.INFO
                else "No action suggested"
            ),
        )

        trace = FailureTrace(
            analyzer_execution_order=("DefaultFailureAnalyzer",),
            diagnostic_sequence=("check_retrieval", "check_verification"),
            classification_path=(category.value,),
            inspected_artifacts=("evidence_bundle", "verification_result"),
            execution_metadata={"verification_result": verification_result},
        )

        from src.core.failure_analysis.failure_analysis_models import (
            FailureFlag,
            FailureMetadata,
        )
        from src.core.failure_analysis.failure_analysis_models import (
            FailureSeverity as LegacySeverity,
        )

        legacy_severity_map = {
            FailureSeverity.INFO: LegacySeverity.NONE,
            FailureSeverity.LOW: LegacySeverity.LOW,
            FailureSeverity.MEDIUM: LegacySeverity.MEDIUM,
            FailureSeverity.HIGH: LegacySeverity.HIGH,
            FailureSeverity.CRITICAL: LegacySeverity.CRITICAL,
        }

        legacy_flag = FailureFlag(
            code=root_cause.value,
            description=diag_summary,
        )
        legacy_metadata = FailureMetadata(strategy_id="DefaultFailureAnalyzer")

        return FailureAnalysisResult(
            classification=classification,
            diagnostic=diagnostic,
            trace=trace,
            failure_flags=frozenset({legacy_flag}),
            severity=legacy_severity_map[severity],
            verification_result=verification_result,
            metadata=legacy_metadata,
        )
