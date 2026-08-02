"""Immutable domain models for Failure Explainability & Reporting subsystem (M3.7)."""

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class FailureExplanationDefinition(BaseModel):
    """Immutable configuration tuning options for failure explanation strategies."""

    strategy: str = Field(default="summary")
    verbosity: str = Field(default="standard")
    include_root_cause: bool = Field(default=True)
    include_correlation: bool = Field(default=True)
    include_severity: bool = Field(default=True)
    include_benchmark_references: bool = Field(default=False)

    model_config = ConfigDict(frozen=True)


class FailureExplanationTemplate(BaseModel):
    """Immutable template configuration for formatting explanations."""

    template_id: str = Field(..., min_length=1)
    verbosity: str = Field(default="standard")
    summary_template: str = Field(default="Failure Summary: {summary}")
    detail_template: str = Field(default="Details: {details}")

    model_config = ConfigDict(frozen=True)


class FailureExplanationMetadata(BaseModel):
    """Immutable execution metadata for generated explanations."""

    strategy_id: str = Field(..., min_length=1)
    configuration_fingerprint: str = Field(..., min_length=1)
    schema_version: str = Field(default="1.0")
    benchmark_reference: str | None = Field(default=None)
    generation_timestamp: str = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)


class FailureEvidenceExplanation(BaseModel):
    """Immutable representation of evidence supporting diagnostic failure explanations."""

    supporting_diagnostics: tuple[str, ...] = Field(default_factory=tuple)
    contributing_failures: tuple[str, ...] = Field(default_factory=tuple)
    ignored_failures: tuple[str, ...] = Field(default_factory=tuple)
    evidence_trace: tuple[str, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(frozen=True)


class FailureDecisionTrace(BaseModel):
    """Immutable trace of pipeline decisions culminating in the failure explanation."""

    correlation_path: tuple[str, ...] = Field(default_factory=tuple)
    attribution_path: tuple[str, ...] = Field(default_factory=tuple)
    severity_policy_path: tuple[str, ...] = Field(default_factory=tuple)
    reasoning_chain: tuple[str, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(frozen=True)


class FailureExplanationResult(BaseModel):
    """Immutable failure explanation result aggregating summary, details, evidence, and trace."""

    summary: str = Field(..., min_length=1)
    detailed_explanation: str = Field(default="")
    evidence_explanation: FailureEvidenceExplanation = Field(...)
    decision_trace: FailureDecisionTrace = Field(...)
    metadata: FailureExplanationMetadata = Field(...)

    model_config = ConfigDict(frozen=True)


class FailureExplanationProfile(BaseModel):
    """Immutable pairing of a profile identifier with an explanation definition and strategy."""

    profile_id: str = Field(..., min_length=1)
    definition: FailureExplanationDefinition = Field(...)
    strategy: Any = Field(...)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "FailureExplanationProfile":
        if hasattr(self.strategy, "validate_compatibility"):
            self.strategy.validate_compatibility(self.definition)
        return self


class FailureExplanationProfileRegistry(BaseModel):
    """O(1) registry resolver for failure explanation profiles."""

    profiles: tuple[FailureExplanationProfile, ...] = Field(..., min_length=1)

    _profile_index: dict[str, FailureExplanationProfile] = PrivateAttr(
        default_factory=dict
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "FailureExplanationProfileRegistry":
        from src.core.exceptions import DuplicateFailureAnalysisProfileError

        index: dict[str, FailureExplanationProfile] = {}
        for p in self.profiles:
            if p.profile_id in index:
                raise DuplicateFailureAnalysisProfileError(
                    f"Duplicate failure explanation profile identifier: {p.profile_id}"
                )
            index[p.profile_id] = p
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> FailureExplanationProfile:
        from src.core.exceptions import FailureAnalysisProfileNotFoundError

        if profile_id not in self._profile_index:
            raise FailureAnalysisProfileNotFoundError(
                f"Failure explanation profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]


def compute_explanation_fingerprint(
    definition: FailureExplanationDefinition,
) -> str:
    """Produce a deterministic SHA-256 fingerprint of a FailureExplanationDefinition."""
    canonical = json.dumps(
        {
            "include_benchmark_references": definition.include_benchmark_references,
            "include_correlation": definition.include_correlation,
            "include_root_cause": definition.include_root_cause,
            "include_severity": definition.include_severity,
            "strategy": definition.strategy,
            "verbosity": definition.verbosity,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
