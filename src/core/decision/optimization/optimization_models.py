"""Immutable Pydantic models for Decision Engine Production Optimization & Hardening (M4.7)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class DecisionCacheDefinition(BaseModel):
    """Immutable cache configuration settings."""

    enabled: bool = Field(default=True)
    max_size: int = Field(default=1000, gt=0)
    ttl_seconds: int = Field(default=300, gt=0)

    model_config = ConfigDict(frozen=True)


class DecisionExecutionGuardDefinition(BaseModel):
    """Immutable execution guard and retry settings."""

    timeout_ms: int = Field(default=1000, gt=0)
    max_retries: int = Field(default=3, ge=0)
    fallback_action: str = Field(default="ABSTAIN", min_length=1)

    model_config = ConfigDict(frozen=True)


class DecisionOptimizationDefinition(BaseModel):
    """Unified configuration for cache and execution guard."""

    cache_config: DecisionCacheDefinition = Field(
        default_factory=DecisionCacheDefinition
    )
    guard_config: DecisionExecutionGuardDefinition = Field(
        default_factory=DecisionExecutionGuardDefinition
    )

    model_config = ConfigDict(frozen=True)


class DecisionExecutionMetrics(BaseModel):
    """Immutable execution statistics for auditability and observability."""

    decision_latency_ms: float = Field(..., ge=0.0)
    cache_hit: bool = Field(...)
    fallback_used: bool = Field(...)
    evaluated_policy_count: int = Field(..., ge=0)
    total_execution_ms: float = Field(..., ge=0.0)

    model_config = ConfigDict(frozen=True)


class DecisionOptimizationProfile(BaseModel):
    """Profile linking an optimization configuration to a profile ID."""

    profile_id: str = Field(..., min_length=1)
    definition: DecisionOptimizationDefinition = Field(...)

    model_config = ConfigDict(frozen=True)


class DecisionOptimizationProfileRegistry(BaseModel):
    """Registry mapping optimization profile IDs, validating duplicates and compatibility."""

    profiles: tuple[DecisionOptimizationProfile, ...] = Field(..., min_length=1)

    _profile_index: dict[str, DecisionOptimizationProfile] = PrivateAttr(
        default_factory=dict
    )

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "DecisionOptimizationProfileRegistry":
        from src.core.exceptions import DuplicateDecisionOptimizationProfileError

        index: dict[str, DecisionOptimizationProfile] = {}
        for p in self.profiles:
            profile_id = p.profile_id
            if profile_id in index:
                raise DuplicateDecisionOptimizationProfileError(
                    f"Duplicate optimization profile ID detected: {profile_id}"
                )
            index[profile_id] = p
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> DecisionOptimizationProfile:
        from src.core.exceptions import DecisionOptimizationProfileNotFoundError

        if profile_id not in self._profile_index:
            raise DecisionOptimizationProfileNotFoundError(
                f"Decision optimization profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]

    def validate_compatibility(self, definition: Any) -> None:
        """Validates optimization profile registry parameters."""
        pass
