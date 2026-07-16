"""Immutable models representing dataset documentation and manifests."""

import re
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.datasets.artifact_models import ArtifactIdentity
from src.core.exceptions import UnsupportedSchemaVersionError


class SupportedTask(str, Enum):
    """Standardized NLP/ML tasks natively supported by datasets."""

    QA = "qa"
    NLI = "nli"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CLASSIFICATION = "classification"
    NER = "ner"
    LM = "language_modeling"


class Language(str, Enum):
    """Standardized ISO 639-1 language codes."""

    EN = "en"
    FR = "fr"
    DE = "de"
    ES = "es"
    ZH = "zh"
    JA = "ja"
    AR = "ar"
    RU = "ru"
    MULTILINGUAL = "multilingual"


class ManifestSchemaVersion(BaseModel):
    """
    Immutable value object encapsulating the semantic version string of the manifest format.
    """

    version: str = Field(
        ..., description="The semantic version string (e.g., '1.0.0')."
    )

    model_config = ConfigDict(frozen=True)

    @field_validator("version")
    def _validate_format(cls, v: str) -> str:
        # Very basic semantic versioning check (e.g., x.y.z)
        if not re.match(r"^\d+\.\d+\.\d+$", v):
            raise UnsupportedSchemaVersionError(f"Invalid schema version format: {v}")
        return v

    @property
    def major(self) -> int:
        return int(self.version.split(".")[0])

    @property
    def minor(self) -> int:
        return int(self.version.split(".")[1])

    @property
    def patch(self) -> int:
        return int(self.version.split(".")[2])


class ArtifactInventoryEntry(BaseModel):
    """
    Extensible structure representing a physical file in the artifact.
    """

    filename: str = Field(..., description="The physical filename.")
    file_type: str = Field(
        ..., description="The type/format of the file (e.g., jsonl, txt)."
    )
    role: str = Field(
        ...,
        description="The role of the file (e.g., train_data, raw_source, metadata).",
    )
    description: str = Field(
        ..., description="Human-readable description of what this file contains."
    )

    model_config = ConfigDict(frozen=True)


class DatasetManifest(BaseModel):
    """
    Immutable, dataset-agnostic documentation contract defining the human and structural
    metadata of a dataset artifact.
    """

    dataset_name: str = Field(..., description="Human-readable title of the dataset.")
    identity: ArtifactIdentity = Field(..., description="The formal identity.")
    description: str = Field(..., description="Comprehensive human-readable summary.")
    source: str = Field(
        ..., description="Original author, organization, or upstream URL."
    )
    license: str = Field(..., description="Explicit legal license.")
    citation: str | None = Field(
        default=None, description="Academic or formal citation instructions."
    )
    supported_tasks: tuple[SupportedTask, ...] = Field(
        ..., description="NLP/ML tasks natively supported."
    )
    languages: tuple[Language, ...] = Field(
        ..., description="Supported ISO language codes."
    )
    domain: str = Field(..., description="General field or category.")
    dataset_structure_summary: str = Field(
        ..., description="High-level description of the expected data shape."
    )
    expected_splits: tuple[str, ...] = Field(
        ..., description="Defined list of available splits."
    )
    artifact_inventory: tuple[ArtifactInventoryEntry, ...] = Field(
        ..., description="List of physical files comprising this version."
    )
    creation_timestamp: datetime = Field(
        ..., description="UTC timestamp denoting when this documentation was generated."
    )
    schema_version: ManifestSchemaVersion = Field(
        ..., description="Version of the manifest schema."
    )

    model_config = ConfigDict(frozen=True)
