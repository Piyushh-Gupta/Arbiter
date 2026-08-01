"""Immutable domain models for Verification Failure Analysis Modernization (M3.5)."""

from enum import Enum
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from src.core.failure_analysis.failure_analysis_models import (
    FailureAnalysisResult as LegacyFailureAnalysisResult,
)


class FailureSeverity(str, Enum):
    """Immutable fail severity levels."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FailureCategory(str, Enum):
    """Concise closed vocabulary of failure categories."""

    RETRIEVAL = "RETRIEVAL"
    VERIFICATION = "VERIFICATION"
    AGGREGATION = "AGGREGATION"
    CALIBRATION = "CALIBRATION"
    EXPLAINABILITY = "EXPLAINABILITY"
    OPTIMIZATION = "OPTIMIZATION"
    CONFIGURATION = "CONFIGURATION"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    UNKNOWN = "UNKNOWN"


class FailureRootCause(str, Enum):
    """Diagnostic root cause classifications."""

    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    LOW_RETRIEVAL_RECALL = "LOW_RETRIEVAL_RECALL"
    CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    CALIBRATION_FAILURE = "CALIBRATION_FAILURE"
    CONFIGURATION_FAILURE = "CONFIGURATION_FAILURE"
    TIMEOUT = "TIMEOUT"
    RESOURCE_EXHAUSTION = "RESOURCE_EXHAUSTION"
    UNKNOWN = "UNKNOWN"


class FailureClassification(BaseModel):
    """Immutable classification details of a failure."""

    category: FailureCategory = Field(...)
    severity: FailureSeverity = Field(...)
    affected_subsystem: str = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)


class FailureArtifactReference(BaseModel):
    """Immutable reference to an inspected pipeline artifact without owning its mutable state."""

    artifact_id: str = Field(..., min_length=1)
    artifact_type: str = Field(..., min_length=1)
    subsystem: str = Field(..., min_length=1)
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class FailureRuntimeMetadata(BaseModel):
    """Immutable metadata tracking analyzer version and execution environment."""

    analyzer_id: str = Field(..., min_length=1)
    analyzer_version: str = Field(..., min_length=1)
    execution_environment: str = Field(..., min_length=1)
    execution_device: str = Field(..., min_length=1)
    framework: str = Field(..., min_length=1)
    execution_timestamp: str = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)


class FailureExecutionMetadata(BaseModel):
    """Immutable metadata tracking request ID and diagnostic latency."""

    request_id: str = Field(..., min_length=1)
    execution_duration: float = Field(..., ge=0.0)
    analyzer_profile: str = Field(..., min_length=1)
    configuration_fingerprint: str = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)


class FailureDiagnostic(BaseModel):
    """Immutable root-cause analysis details."""

    root_cause: FailureRootCause = Field(...)
    diagnostic_summary: str = Field(..., min_length=1)
    affected_artifacts: tuple[str, ...] = Field(default_factory=tuple)
    recovery_recommendation: str = Field(default="No action suggested")

    model_config = ConfigDict(frozen=True)


class DiagnosticEvidence(BaseModel):
    """Immutable details of a specific piece of evidence detected during analysis."""

    analyzer_id: str = Field(..., min_length=1)
    artifact_reference: FailureArtifactReference | Any = Field(...)
    detected_issue: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class AnalyzerExecutionResult(BaseModel):
    """Immutable output from a single specialized analyzer run."""

    analyzer_id: str = Field(..., min_length=1)
    execution_order: int = Field(..., ge=0)
    classification: FailureClassification = Field(...)
    diagnostic_evidence: tuple[DiagnosticEvidence, ...] = Field(default_factory=tuple)
    runtime_metadata: FailureRuntimeMetadata = Field(...)

    model_config = ConfigDict(frozen=True)


class FailureDiagnosticContext(BaseModel):
    """Immutable context separating root-cause analysis from category classification."""

    ordered_analyzer_outputs: tuple[AnalyzerExecutionResult, ...] = Field(
        default_factory=tuple
    )
    aggregated_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class FailureTrace(BaseModel):
    """Immutable audit trace of the diagnosis execution path."""

    analyzer_execution_order: tuple[str, ...] = Field(..., min_length=1)
    diagnostic_sequence: tuple[str, ...] = Field(default_factory=tuple)
    classification_path: tuple[str, ...] = Field(default_factory=tuple)
    inspected_artifacts: tuple[str, ...] = Field(default_factory=tuple)
    execution_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class FailureAnalysisDefinition(BaseModel):
    """Immutable configuration tuning options for failure analyzer strategy."""

    enabled_analyzers: tuple[str, ...] = Field(default_factory=tuple)
    severity_policy: dict[str, FailureSeverity] = Field(default_factory=dict)
    attribution_policy: str = Field(default="default")
    verbosity: str = Field(default="standard")

    model_config = ConfigDict(frozen=True)


class FailureAnalysisInput(BaseModel):
    """Canonical execution contract for failure analysis input parameters."""

    claim: str = Field(..., min_length=1)
    pipeline_artifacts: Mapping[str, FailureArtifactReference | Any] = Field(
        ..., min_length=1
    )
    definition: FailureAnalysisDefinition = Field(...)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class FailureAnalysisResult(LegacyFailureAnalysisResult):
    """Immutable output from the failure analysis execution pipeline."""

    classification: FailureClassification = Field(...)
    diagnostic: FailureDiagnostic = Field(...)
    trace: FailureTrace = Field(...)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class FailureAnalysisProfile(BaseModel):
    """Immutable pairing of a profile identifier and its definition."""

    profile_id: str = Field(..., min_length=1)
    definition: FailureAnalysisDefinition = Field(...)
    analyzer: Any = Field(...)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "FailureAnalysisProfile":
        # 1. Analyzer compatibility validation
        if hasattr(self.analyzer, "validate_compatibility"):
            self.analyzer.validate_compatibility(self.definition)

        # 2. Supported failure categories check
        categories = getattr(self.analyzer, "supported_categories", None)
        if categories is not None:
            for c in categories:
                if not isinstance(c, FailureCategory):
                    raise ValueError(f"Invalid FailureCategory: {c}")

        # 3. Metadata compatibility check
        metadata = getattr(self.analyzer, "runtime_metadata", None)
        if metadata is not None:
            if not isinstance(metadata, FailureRuntimeMetadata):
                raise ValueError(
                    "Analyzer runtime_metadata must be a FailureRuntimeMetadata instance."
                )

        return self


class FailureAnalysisProfileRegistry(BaseModel):
    """O(1) registry resolver for failure analysis profiles."""

    profiles: tuple[FailureAnalysisProfile, ...] = Field(..., min_length=1)

    _profile_index: dict[str, FailureAnalysisProfile] = PrivateAttr(
        default_factory=dict
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "FailureAnalysisProfileRegistry":
        from src.core.exceptions import DuplicateFailureAnalysisProfileError

        index: dict[str, FailureAnalysisProfile] = {}
        for p in self.profiles:
            if p.profile_id in index:
                raise DuplicateFailureAnalysisProfileError(
                    f"Duplicate failure analysis profile identifier: {p.profile_id}"
                )
            index[p.profile_id] = p
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> FailureAnalysisProfile:
        from src.core.exceptions import FailureAnalysisProfileNotFoundError

        if profile_id not in self._profile_index:
            raise FailureAnalysisProfileNotFoundError(
                f"Failure analysis profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]


# --- Failure Correlation Subsystem Models (M3.4) ---


class FailureCorrelationDefinition(BaseModel):
    """Immutable configuration options for failure correlation engine."""

    enabled_strategies: tuple[str, ...] = Field(default_factory=tuple)
    maximum_graph_depth: int = Field(default=5, ge=1)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    grouping_policy: str = Field(default="default")

    model_config = ConfigDict(frozen=True)


class FailureCorrelationRule(BaseModel):
    """Immutable data-driven rule defining a dependency relationship between failure categories."""

    rule_id: str = Field(..., min_length=1)
    source_category: FailureCategory = Field(...)
    target_category: FailureCategory = Field(...)
    precedence: int = Field(default=1, ge=1)
    enabled: bool = Field(default=True)

    model_config = ConfigDict(frozen=True)


class FailureCorrelation(BaseModel):
    """Immutable representation of a single directed correlation edge between failure occurrences."""

    correlation_id: str = Field(..., min_length=1)
    source_failure: str = Field(..., min_length=1)
    target_failure: str = Field(..., min_length=1)
    correlation_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=True)


class FailureCorrelationContext(BaseModel):
    """Immutable context carried into the failure correlation pipeline."""

    analyzer_execution_results: tuple[AnalyzerExecutionResult, ...] = Field(
        default_factory=tuple
    )
    correlation_rules: tuple[FailureCorrelationRule, ...] = Field(default_factory=tuple)
    execution_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class FailureCorrelationResult(BaseModel):
    """Immutable correlation output containing the dependency DAG representation."""

    correlation_graph: tuple[FailureCorrelation, ...] = Field(default_factory=tuple)
    root_failures: tuple[str, ...] = Field(default_factory=tuple)
    dependency_edges: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    summary: str = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)


class FailureCorrelationProfile(BaseModel):
    """Immutable configuration binding a profile identifier with definitions, rules, and strategy."""

    profile_id: str = Field(..., min_length=1)
    definition: FailureCorrelationDefinition = Field(...)
    rules: tuple[FailureCorrelationRule, ...] = Field(default_factory=tuple)
    strategy: Any = Field(...)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "FailureCorrelationProfile":
        if hasattr(self.strategy, "validate_compatibility"):
            self.strategy.validate_compatibility(self.definition)
        return self


class FailureCorrelationProfileRegistry(BaseModel):
    """O(1) registry resolver for failure correlation profiles."""

    profiles: tuple[FailureCorrelationProfile, ...] = Field(..., min_length=1)

    _profile_index: dict[str, FailureCorrelationProfile] = PrivateAttr(
        default_factory=dict
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "FailureCorrelationProfileRegistry":
        from src.core.exceptions import DuplicateFailureAnalysisProfileError

        index: dict[str, FailureCorrelationProfile] = {}
        for p in self.profiles:
            if p.profile_id in index:
                raise DuplicateFailureAnalysisProfileError(
                    f"Duplicate correlation profile identifier: {p.profile_id}"
                )
            # Duplicate rule checks within each profile
            rule_ids = [r.rule_id for r in p.rules]
            if len(rule_ids) != len(set(rule_ids)):
                raise DuplicateFailureAnalysisProfileError(
                    f"Duplicate rule identifier inside profile {p.profile_id}"
                )

            index[p.profile_id] = p
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> FailureCorrelationProfile:
        from src.core.exceptions import FailureAnalysisProfileNotFoundError

        if profile_id not in self._profile_index:
            raise FailureAnalysisProfileNotFoundError(
                f"Failure correlation profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]


# --- Root Cause Attribution & Severity Policy Models (M3.5) ---


class RootCauseAttributionDefinition(BaseModel):
    """Immutable configuration driving the root cause attribution engine."""

    enabled_strategies: tuple[str, ...] = Field(default_factory=tuple)
    traversal_priority: tuple[str, ...] = Field(
        default=(
            "INFRASTRUCTURE",
            "RETRIEVAL",
            "VERIFICATION",
            "CALIBRATION",
            "EXPLAINABILITY",
            "OPTIMIZATION",
            "UNKNOWN",
        )
    )
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=True)


class RootCauseResult(BaseModel):
    """Immutable output of root cause attribution over a failure correlation graph."""

    primary_root_cause: str = Field(..., min_length=1)
    contributing_failures: tuple[str, ...] = Field(default_factory=tuple)
    dependency_path: tuple[str, ...] = Field(default_factory=tuple)
    attribution_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    attribution_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class SeverityRule(BaseModel):
    """Immutable policy rule mapping a failure category to a severity outcome."""

    rule_id: str = Field(..., min_length=1)
    category: FailureCategory = Field(...)
    minimum_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    severity: FailureSeverity = Field(...)
    escalation_required: bool = Field(default=False)
    priority: int = Field(default=1, ge=1)

    model_config = ConfigDict(frozen=True)


class SeverityPolicyDefinition(BaseModel):
    """Immutable configuration for a severity policy engine backed by ordered SeverityRules."""

    rules: tuple[SeverityRule, ...] = Field(default_factory=tuple)
    category_overrides: dict[str, FailureSeverity] = Field(default_factory=dict)
    default_severity: FailureSeverity = Field(default=FailureSeverity.INFO)

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _validate_rule_ids(self) -> "SeverityPolicyDefinition":
        rule_ids = [r.rule_id for r in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError(
                "SeverityPolicyDefinition contains duplicate rule_id values."
            )
        return self


class SeverityEvaluationResult(BaseModel):
    """Immutable output of severity policy evaluation over a root cause result."""

    overall_severity: FailureSeverity = Field(...)
    contributing_severities: tuple[FailureSeverity, ...] = Field(default_factory=tuple)
    escalation_required: bool = Field(default=False)
    escalation_reason: str = Field(default="")
    applied_rule: str | None = Field(default=None)
    policy_trace: tuple[str, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(frozen=True)


class RootCauseProfile(BaseModel):
    """Immutable pairing of a profile identifier with a root cause definition and strategy."""

    profile_id: str = Field(..., min_length=1)
    definition: RootCauseAttributionDefinition = Field(...)
    strategy: Any = Field(...)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "RootCauseProfile":
        if hasattr(self.strategy, "validate_compatibility"):
            self.strategy.validate_compatibility(self.definition)
        return self


class RootCauseProfileRegistry(BaseModel):
    """O(1) registry resolver for root cause attribution profiles."""

    profiles: tuple[RootCauseProfile, ...] = Field(..., min_length=1)

    _profile_index: dict[str, RootCauseProfile] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "RootCauseProfileRegistry":
        from src.core.exceptions import DuplicateFailureAnalysisProfileError

        index: dict[str, RootCauseProfile] = {}
        for p in self.profiles:
            if p.profile_id in index:
                raise DuplicateFailureAnalysisProfileError(
                    f"Duplicate root cause profile identifier: {p.profile_id}"
                )
            index[p.profile_id] = p
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> RootCauseProfile:
        from src.core.exceptions import FailureAnalysisProfileNotFoundError

        if profile_id not in self._profile_index:
            raise FailureAnalysisProfileNotFoundError(
                f"Root cause profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]


class SeverityPolicyProfile(BaseModel):
    """Immutable pairing of a profile identifier with a severity policy definition and policy."""

    profile_id: str = Field(..., min_length=1)
    definition: SeverityPolicyDefinition = Field(...)
    policy: Any = Field(...)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "SeverityPolicyProfile":
        if hasattr(self.policy, "validate_compatibility"):
            self.policy.validate_compatibility(self.definition)
        return self


class SeverityPolicyRegistry(BaseModel):
    """O(1) registry resolver for severity policy profiles."""

    profiles: tuple[SeverityPolicyProfile, ...] = Field(..., min_length=1)

    _profile_index: dict[str, SeverityPolicyProfile] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "SeverityPolicyRegistry":
        from src.core.exceptions import DuplicateFailureAnalysisProfileError

        index: dict[str, SeverityPolicyProfile] = {}
        for p in self.profiles:
            if p.profile_id in index:
                raise DuplicateFailureAnalysisProfileError(
                    f"Duplicate severity policy profile identifier: {p.profile_id}"
                )
            index[p.profile_id] = p
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> SeverityPolicyProfile:
        from src.core.exceptions import FailureAnalysisProfileNotFoundError

        if profile_id not in self._profile_index:
            raise FailureAnalysisProfileNotFoundError(
                f"Severity policy profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]
