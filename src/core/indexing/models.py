"""Immutable domain models for the Offline Indexing Framework."""

from datetime import UTC, datetime
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class Chunk(BaseModel):
    """Immutable value object representing a document chunk during the offline indexing pipeline."""

    span_id: str = Field(
        ...,
        description="Stable identifier for the chunk.",
    )
    document_id: str = Field(
        ...,
        description="Stable identifier for the source document.",
    )
    text: str = Field(
        ...,
        description="Raw chunk text.",
    )
    start_char: int = Field(
        ...,
        description="Starting character offset of this chunk in the original document text.",
    )
    end_char: int = Field(
        ...,
        description="Ending character offset of this chunk in the original document text.",
    )
    dataset_version: str = Field(
        ...,
        description="The checksum/version of the corpus this chunk originated from.",
    )
    metadata: Mapping[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional corpus-specific metadata.",
    )

    model_config = ConfigDict(frozen=True)


class ArtifactLocation(BaseModel):
    """Immutable value object for tracking artifacts and their checksums."""

    path: str
    checksum: str

    model_config = ConfigDict(frozen=True)


def _now_utc() -> datetime:
    return datetime.now(UTC)


class EmbeddingModelMetadata(BaseModel):
    """Immutable metadata describing the embedding model used for dense retrieval."""

    model_id: str = Field(
        ..., description="Identifier of the model used for dense encoding."
    )
    embedding_dimension: int = Field(
        ..., description="The size of the output dense embeddings."
    )
    pooling_strategy: str = Field(
        ..., description="The pooling strategy used (e.g., 'mean', 'cls')."
    )
    normalization_strategy: str = Field(
        ..., description="The normalization strategy used (e.g., 'l2')."
    )
    model_revision: str | None = Field(
        default=None, description="Optional revision or version of the model."
    )

    model_config = ConfigDict(frozen=True)


class IndexManifest(BaseModel):
    """Immutable versioned artifact containing complete metadata about an offline index build."""

    schema_version: str = Field(
        default="1.0.0",
        description="Version of the manifest schema.",
    )
    dataset_version: str = Field(
        ...,
        description="Checksum of the source corpus, indicating the dataset version.",
    )
    embedding_metadata: EmbeddingModelMetadata = Field(
        ..., description="Metadata describing the embedding model used."
    )
    artifacts: dict[str, ArtifactLocation] = Field(
        ...,
        description="Mapping of logical artifact names (e.g. 'dense_index', 'metadata') to their paths and checksums.",
    )
    build_timestamp: datetime = Field(
        default_factory=_now_utc,
        description="UTC timestamp of the index build.",
    )

    model_config = ConfigDict(frozen=True)
