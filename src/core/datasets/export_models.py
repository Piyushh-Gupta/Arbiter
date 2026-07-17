"""Immutable models for the Dataset Export subsystem."""

from __future__ import annotations

import typing
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

if typing.TYPE_CHECKING:
    from src.core.datasets.export.base import BaseExporter
else:
    BaseExporter = typing.Any


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
