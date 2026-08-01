"""Concrete verification failure analysis strategies (M3.2)."""

import warnings
from datetime import datetime
from typing import Any

from src.core.exceptions import FailureAnalysisConfigurationError
from src.core.failure.base import BaseFailureAnalyzer
from src.core.failure.failure_models import (
    FailureAnalysisDefinition,
    FailureAnalysisInput,
    FailureAnalysisResult,
    FailureCategory,
    FailureClassification,
    FailureDiagnostic,
    FailureRootCause,
    FailureRuntimeMetadata,
    FailureSeverity,
    FailureTrace,
)


class DefaultFailureAnalyzer(BaseFailureAnalyzer):
    """Composably executes failure checks across evidence retrieval and verification outcomes."""

    @property
    def supported_categories(self) -> tuple[FailureCategory, ...]:
        """Returns the failure categories supported by this analyzer."""
        return (
            FailureCategory.RETRIEVAL,
            FailureCategory.VERIFICATION,
            FailureCategory.AGGREGATION,
            FailureCategory.UNKNOWN,
        )

    @property
    def runtime_metadata(self) -> FailureRuntimeMetadata:
        """Returns runtime execution provenance for reproducible diagnostics."""
        return FailureRuntimeMetadata(
            analyzer_id="default_failure_analyzer",
            analyzer_version="1.1.0",
            execution_environment="production",
            execution_device="cpu",
            framework="python",
            execution_timestamp=datetime.utcnow().isoformat(),
        )

    def validate_compatibility(self, definition: FailureAnalysisDefinition) -> None:
        if not isinstance(definition, FailureAnalysisDefinition):
            raise FailureAnalysisConfigurationError(
                "DefaultFailureAnalyzer requires FailureAnalysisDefinition."
            )

    def analyze(
        self,
        claim_or_input: str | FailureAnalysisInput,
        verification_result: Any = None,
        definition: Any = None,
    ) -> FailureAnalysisResult:
        if isinstance(claim_or_input, FailureAnalysisInput):
            return self._analyze_canonical(claim_or_input)

        # Deprecated Legacy Adapter path
        warnings.warn(
            "The analyze(claim, verification_result, definition) signature is deprecated "
            "and will be removed in a future release. Use analyze(FailureAnalysisInput) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        input_data = FailureAnalysisInput(
            claim=claim_or_input,
            pipeline_artifacts={"verification_result": verification_result},
            definition=definition,
        )
        return self._analyze_canonical(input_data)

    def _analyze_canonical(
        self, input_data: FailureAnalysisInput
    ) -> FailureAnalysisResult:
        """Canonical execution strategy for FailureAnalysisInput."""
        category = FailureCategory.UNKNOWN
        severity = FailureSeverity.INFO
        root_cause = FailureRootCause.UNKNOWN
        diag_summary = "Diagnosis completed with no issues found."

        # Extract artifacts safely
        verification_result = input_data.pipeline_artifacts.get("verification_result")

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

        from typing import cast

        from src.core.verification.verification_models import VerificationResult

        return FailureAnalysisResult(
            classification=classification,
            diagnostic=diagnostic,
            trace=trace,
            failure_flags=frozenset({legacy_flag}),
            severity=legacy_severity_map[severity],
            verification_result=cast(VerificationResult, verification_result),
            metadata=legacy_metadata,
        )
