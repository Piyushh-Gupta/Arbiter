"""Profile models for the Pipeline subsystem."""

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from src.core.exceptions import (
    DuplicatePipelineProfileError,
    PipelineConfigurationError,
    PipelineProfileNotFoundError,
)
from src.core.pipeline.base import BasePipelineOrchestrator, BasePipelineStage
from src.core.pipeline.pipeline_models import (
    PipelineDefinition,
    PipelineStageDefinition,
)


class PipelineStageProfile(BaseModel):
    """Immutable binding of a PipelineStageDefinition to a concrete stage strategy."""

    profile_id: str = Field(..., min_length=1)
    definition: PipelineStageDefinition = Field(...)
    stage: BasePipelineStage = Field(...)
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "PipelineStageProfile":
        try:
            self.stage.validate_compatibility(self.definition)
        except PipelineConfigurationError:
            raise
        except Exception as e:
            raise PipelineConfigurationError(
                f"Stage {self.profile_id} is incompatible with definition: {e}"
            ) from e
        return self


class PipelineStageRegistry(BaseModel):
    """Registry providing O(1) pipeline stage profile resolution."""

    profiles: tuple[PipelineStageProfile, ...] = Field(..., min_length=1)
    _profile_index: dict[str, PipelineStageProfile] = PrivateAttr(default_factory=dict)
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_index(self) -> "PipelineStageRegistry":
        index = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicatePipelineProfileError(
                    f"Duplicate stage profile_id: {profile.profile_id}"
                )
            index[profile.profile_id] = profile
        self._profile_index = index
        return self

    def resolve(self, profile_id: str) -> PipelineStageProfile:
        if profile_id not in self._profile_index:
            raise PipelineProfileNotFoundError(f"Stage profile {profile_id} not found.")
        return self._profile_index[profile_id]


class PipelineProfile(BaseModel):
    """Immutable binding of a PipelineDefinition to a concrete orchestrator strategy."""

    profile_id: str = Field(..., min_length=1)
    definition: PipelineDefinition = Field(...)
    orchestrator: BasePipelineOrchestrator = Field(...)
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "PipelineProfile":
        try:
            self.orchestrator.validate_compatibility(self.definition)
        except PipelineConfigurationError:
            raise
        except Exception as e:
            raise PipelineConfigurationError(
                f"Pipeline {self.profile_id} is incompatible with definition: {e}"
            ) from e
        return self


class PipelineProfileRegistry(BaseModel):
    """Registry providing O(1) pipeline profile resolution."""

    profiles: tuple[PipelineProfile, ...] = Field(..., min_length=1)
    _profile_index: dict[str, PipelineProfile] = PrivateAttr(default_factory=dict)
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_index(self) -> "PipelineProfileRegistry":
        index = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicatePipelineProfileError(
                    f"Duplicate pipeline profile_id: {profile.profile_id}"
                )
            index[profile.profile_id] = profile
        self._profile_index = index
        return self

    def resolve(self, profile_id: str) -> PipelineProfile:
        if profile_id not in self._profile_index:
            raise PipelineProfileNotFoundError(
                f"Pipeline profile {profile_id} not found."
            )
        return self._profile_index[profile_id]
