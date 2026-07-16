"""Immutable models representing dataset versions and artifact lifecycle states."""

import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.exceptions import InvalidIdentityError


class ArtifactLifecycleState(str, Enum):
    """
    Deterministically derived lifecycle states for dataset artifacts.
    """

    UNINITIALIZED = "UNINITIALIZED"
    INCOMPLETE = "INCOMPLETE"
    DOWNLOADED = "DOWNLOADED"
    READY = "READY"
    CORRUPTED = "CORRUPTED"


class ArtifactIdentity(BaseModel):
    """
    Canonical identity for a dataset artifact enforcing the dataset_id@version structure.
    """

    dataset_id: str = Field(..., description="The underlying dataset identity.")
    version: str = Field(..., description="The explicitly defined version string.")

    model_config = ConfigDict(frozen=True)

    @field_validator("version")
    def _validate_version_pattern(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9.-]+$", v):
            raise InvalidIdentityError(
                f"Version string '{v}' contains invalid characters. Must be alphanumeric, dots, or hyphens."
            )
        return v

    @property
    def canonical(self) -> str:
        """Return the canonical dataset_id@version string."""
        return f"{self.dataset_id}@{self.version}"


class DatasetArtifact(BaseModel):
    """
    An immutable snapshot of a resolved dataset artifact on disk.
    """

    identity: ArtifactIdentity = Field(
        ..., description="The canonical identity of the artifact."
    )
    state: ArtifactLifecycleState = Field(
        ..., description="The deterministically derived lifecycle state."
    )
    path: Path = Field(..., description="The resolved isolated storage directory.")

    model_config = ConfigDict(frozen=True)
