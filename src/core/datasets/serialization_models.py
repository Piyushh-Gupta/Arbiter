import typing
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    NonNegativeInt,
    PrivateAttr,
    model_validator,
)

from src.core.exceptions import (
    DuplicateSerializationProfileError,
    SerializationProfileNotFoundError,
)

if typing.TYPE_CHECKING:
    from src.core.datasets.serialization.base import BaseSerializer
else:
    BaseSerializer = typing.Any


class SerializationFormat(str, Enum):
    """Strongly typed enumeration of supported serialization formats."""

    JSONL = "jsonl"


class SerializationDefinition(BaseModel):
    """Base immutable configuration for a serialization target."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class JsonlSerializationDefinition(SerializationDefinition):
    """Immutable configuration for JSONL dataset serialization."""

    output_path: Path = Field(
        ...,
        description="The absolute or relative destination file path.",
    )
    encoding: str = Field(
        default="utf-8",
        description="Text encoding for the output file.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class MetadataSerializationDefinition(SerializationDefinition):
    """Immutable configuration for dataset metadata serialization."""

    output_path: Path = Field(
        ...,
        description="The absolute or relative destination file path for the metadata JSON.",
    )
    metadata: Mapping[str, JsonValue] = Field(
        ...,
        description="The immutable dataset-level deterministic descriptive metadata to persist.",
    )
    encoding: str = Field(
        default="utf-8",
        description="Text encoding for the output metadata file.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class DatasetManifest(BaseModel):
    """Immutable typed manifest schema describing a serialized dataset."""

    manifest_version: str = Field(
        default="1.0",
        description="The schema version of this manifest for forward compatibility.",
    )
    serialization_format: SerializationFormat = Field(
        ...,
        description="The strongly typed serialization format used.",
    )
    dataset_id: str | None = Field(
        default=None,
        description="Optional externally supplied dataset identifier.",
    )
    dataset_version: str | None = Field(
        default=None,
        description="Optional externally supplied dataset version.",
    )
    created_at: datetime | None = Field(
        default=None,
        description="Optional caller-supplied datetime. Serialized to ISO-8601 via model_dump(mode='json').",
    )
    record_count: NonNegativeInt | None = Field(
        default=None,
        description="Optional caller-supplied non-negative record count. Rejected at construction if negative.",
    )
    extensions: Mapping[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional caller-defined provenance extensions. Immutable after model construction.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class ManifestSerializationDefinition(SerializationDefinition):
    """Immutable configuration for dataset manifest serialization."""

    output_path: Path = Field(
        ...,
        description="The absolute or relative destination file path for the manifest JSON.",
    )
    manifest: DatasetManifest = Field(
        ...,
        description="The immutable typed dataset manifest to persist.",
    )
    encoding: str = Field(
        default="utf-8",
        description="Text encoding for the output manifest file.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class SerializationStep(BaseModel):
    """Binds a SerializationDefinition to its executable BaseSerializer strategy."""

    definition: SerializationDefinition = Field(
        ...,
        description="The strictly immutable configuration for this serialization step.",
    )
    strategy: BaseSerializer = Field(
        ...,
        description="The stateless executable strategy resolving the definition.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "SerializationStep":
        """Statically verifies compatibility between the definition and strategy."""
        self.strategy.validate_compatibility(self.definition)
        return self


class SerializationPipeline(BaseModel):
    """An ordered sequence of SerializationSteps to execute."""

    steps: tuple[SerializationStep, ...] = Field(
        ...,
        description="The ordered collection of strictly bound serialization steps.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class SerializationProfile(BaseModel):
    """Immutable reusable wrapper representing a pre-configured SerializationPipeline."""

    profile_id: str = Field(
        ..., description="Unique identifier for this serialization profile."
    )
    pipeline: SerializationPipeline = Field(
        ...,
        description="The strictly immutable serialization pipeline executing this profile.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class SerializationProfileRegistry(BaseModel):
    """Immutable namespace for securely resolving named serialization profiles."""

    profiles: tuple[SerializationProfile, ...] = Field(
        ..., description="The collection of registered serialization profiles."
    )

    _profile_index: Mapping[str, SerializationProfile] = PrivateAttr(
        default_factory=dict
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_index(self) -> "SerializationProfileRegistry":
        index = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicateSerializationProfileError(
                    f"Duplicate serialization profile identifier: {profile.profile_id}"
                )
            index[profile.profile_id] = profile

        # Explicit architectural requirement: initialize the private lookup index
        # using object.__setattr__ to bypass Pydantic's frozen constraint.
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> SerializationProfile:
        """Resolves a profile by identifier."""
        if profile_id not in self._profile_index:
            raise SerializationProfileNotFoundError(
                f"Serialization profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]
