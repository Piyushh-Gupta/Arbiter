"""Immutable models for the Dataset Loading subsystem."""

from __future__ import annotations

import typing

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

if typing.TYPE_CHECKING:
    from src.core.datasets.loading.base import BaseLoader
else:
    BaseLoader = typing.Any

from src.core.datasets.mapping_models import TaskRecord
from src.core.datasets.serialization_models import DatasetManifest


class LoadingDefinition(BaseModel):
    """Base immutable configuration for a dataset loading target."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class JsonlLoadingDefinition(LoadingDefinition):
    """Immutable configuration targeting a serialized JSONL dataset artifact."""

    encoding: str = Field(
        default="utf-8",
        description="The text encoding of the target JSONL file.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class ArbiterDataset(BaseModel):
    """Canonical immutable representation of a completely reconstructed dataset."""

    manifest: DatasetManifest = Field(
        ..., description="The dataset's formally documented metadata contract."
    )
    records: tuple[TaskRecord, ...] = Field(
        ...,
        description="The completely parsed, contiguous sequence of dataset records.",
    )

    model_config = ConfigDict(frozen=True)


class LoadingProfile(BaseModel):
    """Immutable reusable wrapper binding a loading definition to its execution strategy."""

    profile_id: str = Field(
        ..., description="Unique identifier for this loading profile."
    )
    definition: LoadingDefinition = Field(
        ...,
        description="The strictly immutable configuration for this loading strategy.",
    )
    strategy: "BaseLoader" = Field(
        ..., description="The stateless executable strategy resolving the definition."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "LoadingProfile":
        """Statically verifies compatibility between the definition and strategy upon construction."""
        self.strategy.validate_compatibility(self.definition)
        return self


class LoadingProfileRegistry(BaseModel):
    """Immutable namespace for securely resolving named loading profiles."""

    profiles: tuple[LoadingProfile, ...] = Field(
        ..., description="The abstract collection of registered loading profiles."
    )

    _profile_index: dict[str, LoadingProfile] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "LoadingProfileRegistry":
        from src.core.exceptions import DuplicateLoadingProfileError

        index: dict[str, LoadingProfile] = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicateLoadingProfileError(
                    f"Duplicate loading profile identifier: {profile.profile_id}"
                )
            index[profile.profile_id] = profile

        # Bypass Pydantic's frozen constraint to initialize the O(1) private lookup table
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> LoadingProfile:
        """Resolves a profile statelessly in O(1) time."""
        from src.core.exceptions import LoadingProfileNotFoundError

        if profile_id not in self._profile_index:
            raise LoadingProfileNotFoundError(
                f"Loading profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]
