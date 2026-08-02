"""Concrete verification failure analysis strategies, composite orchestrator, and aggregation logic (M3.3)."""

import warnings
from datetime import UTC, datetime
from typing import Any, Sequence, cast

from src.core.exceptions import FailureAnalysisConfigurationError
from src.core.failure.base import BaseFailureAnalyzer, FailureAggregationStrategy
from src.core.failure.failure_models import (
    AnalyzerExecutionResult,
    DiagnosticEvidence,
    FailureAnalysisDefinition,
    FailureAnalysisInput,
    FailureAnalysisResult,
    FailureArtifactReference,
    FailureCategory,
    FailureClassification,
    FailureDiagnostic,
    FailureRootCause,
    FailureRuntimeMetadata,
    FailureSeverity,
    FailureTrace,
)
from src.core.verification.verification_models import VerificationResult


class RetrievalFailureAnalyzer(BaseFailureAnalyzer):
    """Diagnoses evidence retrieval stage failures."""

    @property
    def supported_categories(self) -> tuple[FailureCategory, ...]:
        return (FailureCategory.RETRIEVAL, FailureCategory.UNKNOWN)

    @property
    def runtime_metadata(self) -> FailureRuntimeMetadata:
        return FailureRuntimeMetadata(
            analyzer_id="retrieval_failure_analyzer",
            analyzer_version="1.0.0",
            execution_environment="production",
            execution_device="cpu",
            framework="python",
            execution_timestamp=datetime.now(UTC).isoformat(),
        )

    def validate_compatibility(self, definition: FailureAnalysisDefinition) -> None:
        if not isinstance(definition, FailureAnalysisDefinition):
            raise FailureAnalysisConfigurationError(
                "Requires FailureAnalysisDefinition."
            )

    def analyze(self, input_data: FailureAnalysisInput) -> FailureAnalysisResult:
        # Default status
        category = FailureCategory.UNKNOWN
        severity = FailureSeverity.INFO
        root_cause = FailureRootCause.UNKNOWN
        issue = "Retrieval diagnostic check completed."

        verification_result = input_data.pipeline_artifacts.get("verification_result")
        bundle = getattr(verification_result, "evidence_bundle", None)
        passages = getattr(bundle, "passages", ()) if bundle is not None else ()

        if not passages:
            category = FailureCategory.RETRIEVAL
            severity = FailureSeverity.CRITICAL
            root_cause = FailureRootCause.MISSING_EVIDENCE
            issue = "No evidence passages retrieved."

        classification = FailureClassification(
            category=category,
            severity=severity,
            affected_subsystem="retrieval",
        )
        diagnostic = FailureDiagnostic(
            root_cause=root_cause,
            diagnostic_summary=issue,
            affected_artifacts=("evidence_bundle",),
        )
        trace = FailureTrace(
            analyzer_execution_order=("RetrievalFailureAnalyzer",),
            classification_path=(category.value,),
            inspected_artifacts=("evidence_bundle",),
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
        legacy_flag = FailureFlag(code=root_cause.value, description=issue)
        legacy_metadata = FailureMetadata(strategy_id="RetrievalFailureAnalyzer")

        return FailureAnalysisResult(
            classification=classification,
            diagnostic=diagnostic,
            trace=trace,
            failure_flags=frozenset({legacy_flag}),
            severity=legacy_severity_map[severity],
            verification_result=cast(VerificationResult, verification_result),
            metadata=legacy_metadata,
        )


class VerificationFailureAnalyzer(BaseFailureAnalyzer):
    """Diagnoses NLI prediction and aggregation stage failures."""

    @property
    def supported_categories(self) -> tuple[FailureCategory, ...]:
        return (
            FailureCategory.VERIFICATION,
            FailureCategory.AGGREGATION,
            FailureCategory.UNKNOWN,
        )

    @property
    def runtime_metadata(self) -> FailureRuntimeMetadata:
        return FailureRuntimeMetadata(
            analyzer_id="verification_failure_analyzer",
            analyzer_version="1.0.0",
            execution_environment="production",
            execution_device="cpu",
            framework="python",
            execution_timestamp=datetime.now(UTC).isoformat(),
        )

    def validate_compatibility(self, definition: FailureAnalysisDefinition) -> None:
        if not isinstance(definition, FailureAnalysisDefinition):
            raise FailureAnalysisConfigurationError(
                "Requires FailureAnalysisDefinition."
            )

    def analyze(self, input_data: FailureAnalysisInput) -> FailureAnalysisResult:
        category = FailureCategory.UNKNOWN
        severity = FailureSeverity.INFO
        root_cause = FailureRootCause.UNKNOWN
        issue = "Verification diagnostic check completed."

        verification_result = input_data.pipeline_artifacts.get("verification_result")
        if verification_result is not None:
            confidence = getattr(verification_result, "confidence", None)
            if confidence is not None and confidence < 0.5:
                category = FailureCategory.VERIFICATION
                severity = FailureSeverity.HIGH
                root_cause = FailureRootCause.LOW_CONFIDENCE
                issue = f"Low verification confidence: {confidence:.2f} < 0.5."
            else:
                contradicting = getattr(
                    verification_result, "contradicting_passages", ()
                )
                supporting = getattr(verification_result, "supporting_passages", ())
                if contradicting and supporting:
                    category = FailureCategory.AGGREGATION
                    severity = FailureSeverity.MEDIUM
                    root_cause = FailureRootCause.CONTRADICTORY_EVIDENCE
                    issue = "Contradictory evidence detected in passages."

        classification = FailureClassification(
            category=category,
            severity=severity,
            affected_subsystem="verification",
        )
        diagnostic = FailureDiagnostic(
            root_cause=root_cause,
            diagnostic_summary=issue,
            affected_artifacts=("verification_result",),
        )
        trace = FailureTrace(
            analyzer_execution_order=("VerificationFailureAnalyzer",),
            classification_path=(category.value,),
            inspected_artifacts=("verification_result",),
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
        legacy_flag = FailureFlag(code=root_cause.value, description=issue)
        legacy_metadata = FailureMetadata(strategy_id="VerificationFailureAnalyzer")

        return FailureAnalysisResult(
            classification=classification,
            diagnostic=diagnostic,
            trace=trace,
            failure_flags=frozenset({legacy_flag}),
            severity=legacy_severity_map[severity],
            verification_result=cast(VerificationResult, verification_result),
            metadata=legacy_metadata,
        )


class CalibrationFailureAnalyzer(BaseFailureAnalyzer):
    """Diagnoses post-verification confidence calibration stage failures."""

    @property
    def supported_categories(self) -> tuple[FailureCategory, ...]:
        return (FailureCategory.CALIBRATION, FailureCategory.UNKNOWN)

    @property
    def runtime_metadata(self) -> FailureRuntimeMetadata:
        return FailureRuntimeMetadata(
            analyzer_id="calibration_failure_analyzer",
            analyzer_version="1.0.0",
            execution_environment="production",
            execution_device="cpu",
            framework="python",
            execution_timestamp=datetime.now(UTC).isoformat(),
        )

    def validate_compatibility(self, definition: FailureAnalysisDefinition) -> None:
        if not isinstance(definition, FailureAnalysisDefinition):
            raise FailureAnalysisConfigurationError(
                "Requires FailureAnalysisDefinition."
            )

    def analyze(self, input_data: FailureAnalysisInput) -> FailureAnalysisResult:
        category = FailureCategory.UNKNOWN
        severity = FailureSeverity.INFO
        root_cause = FailureRootCause.UNKNOWN
        issue = "Calibration diagnostic check completed."

        calibration_result = input_data.pipeline_artifacts.get("calibration_result")
        if calibration_result is not None:
            calibrated_confidence = getattr(
                calibration_result, "calibrated_confidence", None
            )
            if calibrated_confidence is not None and (
                calibrated_confidence < 0.0 or calibrated_confidence > 1.0
            ):
                category = FailureCategory.CALIBRATION
                severity = FailureSeverity.HIGH
                root_cause = FailureRootCause.CALIBRATION_FAILURE
                issue = "Calibrated confidence value is out of bounds."

        classification = FailureClassification(
            category=category,
            severity=severity,
            affected_subsystem="calibration",
        )
        diagnostic = FailureDiagnostic(
            root_cause=root_cause,
            diagnostic_summary=issue,
            affected_artifacts=("calibration_result",),
        )
        trace = FailureTrace(
            analyzer_execution_order=("CalibrationFailureAnalyzer",),
            classification_path=(category.value,),
            inspected_artifacts=("calibration_result",),
        )

        verification_result = input_data.pipeline_artifacts.get("verification_result")

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
        legacy_flag = FailureFlag(code=root_cause.value, description=issue)
        legacy_metadata = FailureMetadata(strategy_id="CalibrationFailureAnalyzer")

        return FailureAnalysisResult(
            classification=classification,
            diagnostic=diagnostic,
            trace=trace,
            failure_flags=frozenset({legacy_flag}),
            severity=legacy_severity_map[severity],
            verification_result=cast(VerificationResult, verification_result),
            metadata=legacy_metadata,
        )


class InfrastructureFailureAnalyzer(BaseFailureAnalyzer):
    """Diagnoses timeout, resource, and pipeline configuration failures."""

    @property
    def supported_categories(self) -> tuple[FailureCategory, ...]:
        return (
            FailureCategory.INFRASTRUCTURE,
            FailureCategory.CONFIGURATION,
            FailureCategory.UNKNOWN,
        )

    @property
    def runtime_metadata(self) -> FailureRuntimeMetadata:
        return FailureRuntimeMetadata(
            analyzer_id="infrastructure_failure_analyzer",
            analyzer_version="1.0.0",
            execution_environment="production",
            execution_device="cpu",
            framework="python",
            execution_timestamp=datetime.now(UTC).isoformat(),
        )

    def validate_compatibility(self, definition: FailureAnalysisDefinition) -> None:
        if not isinstance(definition, FailureAnalysisDefinition):
            raise FailureAnalysisConfigurationError(
                "Requires FailureAnalysisDefinition."
            )

    def analyze(self, input_data: FailureAnalysisInput) -> FailureAnalysisResult:
        category = FailureCategory.UNKNOWN
        severity = FailureSeverity.INFO
        root_cause = FailureRootCause.UNKNOWN
        issue = "Infrastructure diagnostic check completed."

        # Check for configured/simulated timeouts or duration
        duration = input_data.pipeline_artifacts.get("execution_duration")
        if (
            isinstance(duration, (int, float)) and duration > 5000.0
        ):  # arbitrary 5000ms threshold
            category = FailureCategory.INFRASTRUCTURE
            severity = FailureSeverity.HIGH
            root_cause = FailureRootCause.TIMEOUT
            issue = "Execution exceeded time bounds (Timeout)."

        classification = FailureClassification(
            category=category,
            severity=severity,
            affected_subsystem="infrastructure",
        )
        diagnostic = FailureDiagnostic(
            root_cause=root_cause,
            diagnostic_summary=issue,
            affected_artifacts=(),
        )
        trace = FailureTrace(
            analyzer_execution_order=("InfrastructureFailureAnalyzer",),
            classification_path=(category.value,),
            inspected_artifacts=(),
        )

        verification_result = input_data.pipeline_artifacts.get("verification_result")

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
        legacy_flag = FailureFlag(code=root_cause.value, description=issue)
        legacy_metadata = FailureMetadata(strategy_id="InfrastructureFailureAnalyzer")

        return FailureAnalysisResult(
            classification=classification,
            diagnostic=diagnostic,
            trace=trace,
            failure_flags=frozenset({legacy_flag}),
            severity=legacy_severity_map[severity],
            verification_result=cast(VerificationResult, verification_result),
            metadata=legacy_metadata,
        )


class DefaultFailureAggregationStrategy(FailureAggregationStrategy):
    """Combines specialized AnalyzerExecutionResult sequence based on category and severity precedence rules."""

    def aggregate(
        self,
        results: Sequence[AnalyzerExecutionResult],
        input_data: FailureAnalysisInput,
    ) -> FailureAnalysisResult:
        # Precedence definition tables
        severity_precedence = {
            FailureSeverity.CRITICAL: 5,
            FailureSeverity.HIGH: 4,
            FailureSeverity.MEDIUM: 3,
            FailureSeverity.LOW: 2,
            FailureSeverity.INFO: 1,
        }
        category_precedence = {
            FailureCategory.CONFIGURATION: 9,
            FailureCategory.INFRASTRUCTURE: 8,
            FailureCategory.RETRIEVAL: 7,
            FailureCategory.VERIFICATION: 6,
            FailureCategory.AGGREGATION: 5,
            FailureCategory.CALIBRATION: 4,
            FailureCategory.EXPLAINABILITY: 3,
            FailureCategory.OPTIMIZATION: 2,
            FailureCategory.UNKNOWN: 1,
        }

        selected_severity = FailureSeverity.INFO
        selected_category = FailureCategory.UNKNOWN
        selected_root_cause = FailureRootCause.UNKNOWN
        subsystem = "verification"

        diagnostic_summaries: list[str] = []
        execution_order: list[str] = []
        inspected: list[str] = []

        for r in results:
            execution_order.append(r.analyzer_id)
            for evidence in r.diagnostic_evidence:
                diagnostic_summaries.append(evidence.detected_issue)
                if isinstance(evidence.artifact_reference, FailureArtifactReference):
                    inspected.append(evidence.artifact_reference.artifact_id)

            # Max severity resolution
            if (
                severity_precedence[r.classification.severity]
                > severity_precedence[selected_severity]
            ):
                selected_severity = r.classification.severity

            # Max category resolution
            if (
                category_precedence[r.classification.category]
                > category_precedence[selected_category]
            ):
                selected_category = r.classification.category
                subsystem = r.classification.affected_subsystem

        # Determine Root Cause
        if selected_category == FailureCategory.RETRIEVAL:
            selected_root_cause = FailureRootCause.MISSING_EVIDENCE
        elif selected_category == FailureCategory.VERIFICATION:
            selected_root_cause = FailureRootCause.LOW_CONFIDENCE
        elif selected_category == FailureCategory.AGGREGATION:
            selected_root_cause = FailureRootCause.CONTRADICTORY_EVIDENCE
        elif selected_category == FailureCategory.CALIBRATION:
            selected_root_cause = FailureRootCause.CALIBRATION_FAILURE
        elif selected_category == FailureCategory.INFRASTRUCTURE:
            selected_root_cause = FailureRootCause.TIMEOUT
        elif selected_category == FailureCategory.CONFIGURATION:
            selected_root_cause = FailureRootCause.CONFIGURATION_FAILURE

        final_summary = (
            " | ".join(diagnostic_summaries)
            if diagnostic_summaries
            else "Diagnosis completed with no issues found."
        )

        classification = FailureClassification(
            category=selected_category,
            severity=selected_severity,
            affected_subsystem=subsystem,
        )

        diagnostic = FailureDiagnostic(
            root_cause=selected_root_cause,
            diagnostic_summary=final_summary,
            affected_artifacts=tuple(set(inspected)),
        )

        trace = FailureTrace(
            analyzer_execution_order=tuple(execution_order),
            inspected_artifacts=tuple(set(inspected)),
            execution_metadata={
                "verification_result": input_data.pipeline_artifacts.get(
                    "verification_result"
                )
            },
        )

        # Legacy backward-compatibility mappings
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
            code=selected_root_cause.value, description=final_summary
        )
        legacy_metadata = FailureMetadata(strategy_id="CompositeFailureAnalyzer")

        verification_result = input_data.pipeline_artifacts.get("verification_result")

        return FailureAnalysisResult(
            classification=classification,
            diagnostic=diagnostic,
            trace=trace,
            failure_flags=frozenset({legacy_flag}),
            severity=legacy_severity_map[selected_severity],
            verification_result=cast(VerificationResult, verification_result),
            metadata=legacy_metadata,
        )


class CompositeFailureAnalyzer(BaseFailureAnalyzer):
    """Executes sequence of analyzers and aggregates diagnostics via FailureAggregationStrategy."""

    def __init__(
        self,
        analyzers: Sequence[BaseFailureAnalyzer],
        aggregation_strategy: FailureAggregationStrategy,
    ) -> None:
        self.analyzers = tuple(analyzers)
        self.aggregation_strategy = aggregation_strategy

    @property
    def supported_categories(self) -> tuple[FailureCategory, ...]:
        cats: set[FailureCategory] = set()
        for a in self.analyzers:
            if hasattr(a, "supported_categories"):
                cats.update(a.supported_categories)
        return tuple(cats)

    @property
    def runtime_metadata(self) -> FailureRuntimeMetadata:
        return FailureRuntimeMetadata(
            analyzer_id="composite_failure_analyzer",
            analyzer_version="1.0.0",
            execution_environment="production",
            execution_device="cpu",
            framework="python",
            execution_timestamp=datetime.now(UTC).isoformat(),
        )

    def validate_compatibility(self, definition: FailureAnalysisDefinition) -> None:
        for a in self.analyzers:
            a.validate_compatibility(definition)

    def analyze(
        self,
        claim_or_input: str | FailureAnalysisInput,
        verification_result: Any = None,
        definition: Any = None,
    ) -> FailureAnalysisResult:
        if isinstance(claim_or_input, FailureAnalysisInput):
            return self._analyze_canonical(claim_or_input)

        # Deprecated legacy adapter path
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
        execution_results: list[AnalyzerExecutionResult] = []

        for idx, analyzer in enumerate(self.analyzers):
            # Inspect artifacts and run analyzer
            individual_res = analyzer.analyze(input_data)

            # Extract Diagnostic Evidence
            evidence = DiagnosticEvidence(
                analyzer_id=analyzer.runtime_metadata.analyzer_id,
                artifact_reference=FailureArtifactReference(
                    artifact_id=individual_res.diagnostic.affected_artifacts[0]
                    if individual_res.diagnostic.affected_artifacts
                    else "unknown_artifact",
                    artifact_type="subsystem_outcome",
                    subsystem=individual_res.classification.affected_subsystem,
                ),
                detected_issue=individual_res.diagnostic.diagnostic_summary,
                confidence=1.0,
            )

            execution_results.append(
                AnalyzerExecutionResult(
                    analyzer_id=analyzer.runtime_metadata.analyzer_id,
                    execution_order=idx,
                    classification=individual_res.classification,
                    diagnostic_evidence=(evidence,),
                    runtime_metadata=analyzer.runtime_metadata,
                )
            )

        return self.aggregation_strategy.aggregate(execution_results, input_data)


class DefaultFailureAnalyzer(BaseFailureAnalyzer):
    """Wrapper coordinating M3 specialized diagnostics, maintaining backward compatibility adapter."""

    def __init__(self) -> None:
        # Construct composite components
        self._composite = CompositeFailureAnalyzer(
            analyzers=(
                RetrievalFailureAnalyzer(),
                VerificationFailureAnalyzer(),
                CalibrationFailureAnalyzer(),
                InfrastructureFailureAnalyzer(),
            ),
            aggregation_strategy=DefaultFailureAggregationStrategy(),
        )

    @property
    def supported_categories(self) -> tuple[FailureCategory, ...]:
        return self._composite.supported_categories

    @property
    def runtime_metadata(self) -> FailureRuntimeMetadata:
        return self._composite.runtime_metadata

    def validate_compatibility(self, definition: FailureAnalysisDefinition) -> None:
        self._composite.validate_compatibility(definition)

    def analyze(
        self,
        claim_or_input: str | FailureAnalysisInput,
        verification_result: Any = None,
        definition: Any = None,
    ) -> FailureAnalysisResult:
        if isinstance(claim_or_input, FailureAnalysisInput):
            return self._composite.analyze(claim_or_input)

        # Deprecated legacy adapter path
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
        return self._composite.analyze(input_data)
