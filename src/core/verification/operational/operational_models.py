"""Immutable operational domain models for Verification Production Hardening (M2.9)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class VerificationOperationalProfile(BaseModel):
    """Immutable operational configuration profile containing logging, environment, and readiness rules."""

    profile_id: str = Field(..., min_length=1)
    environment: str = Field(..., min_length=1)
    logging_configuration: dict[str, Any] = Field(default_factory=dict)
    readiness_configuration: dict[str, Any] = Field(default_factory=dict)
    telemetry_configuration: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class VerificationOperationalRegistry(BaseModel):
    """O(1) lookup registry for resolving verification operational profiles."""

    profiles: tuple[VerificationOperationalProfile, ...] = Field(..., min_length=1)

    _profile_index: dict[str, VerificationOperationalProfile] = PrivateAttr(
        default_factory=dict
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "VerificationOperationalRegistry":
        from src.core.exceptions import DuplicateOptimizationProfileError

        index: dict[str, VerificationOperationalProfile] = {}
        for p in self.profiles:
            if p.profile_id in index:
                # Reuse/raise configuration duplication exceptions
                raise DuplicateOptimizationProfileError(
                    f"Duplicate operational profile identifier: {p.profile_id}"
                )
            if p.environment not in ("production", "staging", "development"):
                raise ValueError(
                    f"Incompatible operational profile environment: {p.environment}"
                )
            index[p.profile_id] = p
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> VerificationOperationalProfile:
        from src.core.exceptions import OptimizationProfileNotFoundError

        if profile_id not in self._profile_index:
            raise OptimizationProfileNotFoundError(
                f"Verification operational profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]


class VerificationOperationalTrace(BaseModel):
    """Immutable audit trace capturing operational startup verification results."""

    startup_validation: bool = Field(...)
    readiness_validation: bool = Field(...)
    registry_validation: bool = Field(...)
    operational_configuration: dict[str, Any] = Field(...)
    execution_timestamp: str = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)
