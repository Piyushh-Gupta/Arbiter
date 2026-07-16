"""Data models for the preprocessing layer."""

from __future__ import annotations

import typing
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from src.core.datasets.mapping_models import TaskRecord
from src.core.datasets.partitioning_models import PartitionId
from src.core.exceptions import (
    DuplicatePreprocessingProfileError,
    PreprocessingProfileNotFoundError,
)

if typing.TYPE_CHECKING:
    from src.core.datasets.preprocessing.base import PreprocessingPipeline
else:
    PreprocessingPipeline = typing.Any


class PreprocessingMetadata(BaseModel):
    """Canonical structure storing preprocessing outputs and generic metadata."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


@dataclass(frozen=True)
class PreprocessedRecord:
    """Canonical structure wrapping records with accumulated preprocessing metadata."""

    partition: PartitionId
    record: TaskRecord
    preprocessing_metadata: PreprocessingMetadata


@typing.runtime_checkable
class PreprocessingDefinition(typing.Protocol):
    """Abstract protocol for declarative preprocessing parameters."""

    pass


class PassThroughPreprocessingDefinition(BaseModel):
    """Empty configuration driving the baseline pipeline proof of concept."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class PreprocessingProfile(BaseModel):
    """Reusable infrastructure configuration object grouping dataset transformation patterns natively."""

    profile_id: str = Field(
        ..., description="Unique deterministic identifier mapped universally."
    )
    pipeline: PreprocessingPipeline = Field(
        ..., description="Strictly chained preprocessor sequencing."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class PreprocessingProfileRegistry(BaseModel):
    """Immutable namespace owning protected composite profiles avoiding mapping collisions."""

    profiles: tuple[PreprocessingProfile, ...] = Field(
        ..., description="Abstract sequence mapping protected structures natively."
    )

    _profile_index: Mapping[str, PreprocessingProfile] = PrivateAttr(
        default_factory=dict
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "PreprocessingProfileRegistry":
        index: dict[str, PreprocessingProfile] = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicatePreprocessingProfileError(
                    f"Profile ID '{profile.profile_id}' duplicated."
                )
            index[profile.profile_id] = profile

        # Build index safely
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> PreprocessingProfile:
        """Resolve an existing profile, raising domain exception if unknown."""
        if profile_id not in self._profile_index:
            raise PreprocessingProfileNotFoundError(
                f"Profile ID '{profile_id}' not found."
            )
        return self._profile_index[profile_id]
