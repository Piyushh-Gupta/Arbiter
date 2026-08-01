"""Immutable domain models for Verification Failure Analysis Modernization (M3.2)."""

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


class FailureDiagnosticContext(BaseModel):
    """Immutable context separating root-cause analysis from category classification."""

    ordered_analyzer_outputs: tuple[Any, ...] = Field(default_factory=tuple)
    inspected_artifact_references: tuple[FailureArtifactReference, ...] = Field(
        default_factory=tuple
    )
    execution_metadata: dict[str, Any] = Field(default_factory=dict)

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
