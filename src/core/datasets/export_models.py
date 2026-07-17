"""Immutable models for the Dataset Export subsystem."""

from __future__ import annotations

import typing
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

if typing.TYPE_CHECKING:
    from src.core.datasets.export.base import BaseExporter
else:
    BaseExporter = typing.Any

from src.core.exceptions import DuplicateExportProfileError, ExportProfileNotFoundError


class SerializedArtifact(BaseModel):
    """Immutable representation of a completely serialized dataset artifact."""

    root_path: Path = Field(
        ..., description="The root path of the serialized artifact."
    )

    model_config = ConfigDict(frozen=True)


class ExportDefinition(BaseModel):
    """Base immutable configuration for an export target."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class LocalExportDefinition(ExportDefinition):
    """Immutable configuration for local filesystem export."""

    destination_root: Path = Field(
        ...,
        description="The absolute or relative path where the artifact will be copied.",
    )
    overwrite_existing: bool = Field(
        default=False,
        description="If True, permits overwriting an existing destination directory or file.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class HuggingFaceRepositoryType(str, Enum):
    """Supported Hugging Face Hub repository types."""

    DATASET = "dataset"
    MODEL = "model"
    SPACE = "space"


class HuggingFaceExportDefinition(ExportDefinition):
    """Immutable configuration for Hugging Face Hub export."""

    repository_id: str = Field(
        ...,
        description="The target Hugging Face repository ID (e.g., 'username/dataset').",
    )
    repository_type: HuggingFaceRepositoryType = Field(
        default=HuggingFaceRepositoryType.DATASET,
        description="The strongly typed repository type to target.",
    )
    revision: str = Field(
        default="main",
        description="The target git branch, tag, or commit revision.",
    )
    commit_message: str = Field(
        default="Upload dataset via Arbiter Export Subsystem",
        description="The commit message to associate with the upload.",
    )
    private: bool = Field(
        default=True,
        description="If True, the repository will be private if it is created.",
    )
    create_repo_if_missing: bool = Field(
        default=True,
        description="If True, automatically creates the repository if it does not exist.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class S3StorageClass(str, Enum):
    """Supported Amazon S3 storage classes."""

    STANDARD = "STANDARD"
    REDUCED_REDUNDANCY = "REDUCED_REDUNDANCY"
    STANDARD_IA = "STANDARD_IA"
    ONEZONE_IA = "ONEZONE_IA"
    INTELLIGENT_TIERING = "INTELLIGENT_TIERING"
    GLACIER = "GLACIER"
    DEEP_ARCHIVE = "DEEP_ARCHIVE"
    OUTPOSTS = "OUTPOSTS"
    GLACIER_IR = "GLACIER_IR"
    SNOW = "SNOW"


class S3ExportDefinition(ExportDefinition):
    """Immutable configuration for Amazon S3 export."""

    bucket_name: str = Field(
        ...,
        description="The target Amazon S3 bucket name.",
    )
    object_prefix: str = Field(
        ...,
        description="The S3 key prefix (e.g., 'datasets/my_export'). Will be treated as a virtual directory.",
    )
    region_name: str | None = Field(
        default=None,
        description="Optional AWS region. If None, relies on the standard boto3 environment resolution.",
    )
    storage_class: S3StorageClass | None = Field(
        default=None,
        description="Optional strongly typed S3 storage class. If None, uses bucket default.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class ExportStep(BaseModel):
    """Immutable bound executable defining one exact export destination."""

    definition: ExportDefinition
    strategy: BaseExporter

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "ExportStep":
        self.strategy.validate_compatibility(self.definition)
        return self


class ExportPipeline(BaseModel):
    """Ordered pipeline of configured export steps."""

    steps: tuple[ExportStep, ...]

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class ExportProfile(BaseModel):
    """Immutable named configuration encapsulating an export pipeline."""

    profile_id: str = Field(
        ...,
        description="Unique identifier for the export profile.",
    )
    pipeline: ExportPipeline = Field(
        ...,
        description="The immutable export pipeline wrapped by this profile.",
    )

    model_config = ConfigDict(frozen=True)


class ExportProfileRegistry(BaseModel):
    """Immutable registry of export profiles."""

    profiles: tuple[ExportProfile, ...] = Field(
        ...,
        description="The immutable collection of registered export profiles.",
    )

    _lookup_table: dict[str, ExportProfile] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _build_lookup_table(self) -> "ExportProfileRegistry":
        lookup: dict[str, ExportProfile] = {}
        for profile in self.profiles:
            if profile.profile_id in lookup:
                raise DuplicateExportProfileError(
                    f"Export profile identifier '{profile.profile_id}' is already registered."
                )
            lookup[profile.profile_id] = profile

        object.__setattr__(self, "_lookup_table", lookup)
        return self

    def resolve(self, profile_id: str) -> ExportProfile:
        """Resolves an export profile by its unique identifier."""
        if profile_id not in self._lookup_table:
            raise ExportProfileNotFoundError(
                f"Export profile '{profile_id}' not found in registry."
            )
        return self._lookup_table[profile_id]

    model_config = ConfigDict(frozen=True)
