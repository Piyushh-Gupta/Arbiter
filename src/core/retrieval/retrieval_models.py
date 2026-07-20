"""Immutable domain models for the Evidence Retrieval subsystem."""

from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class RetrievalDefinition(BaseModel):
    """Base immutable configuration for a retrieval strategy."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class CorpusEntry(BaseModel):
    """A minimal immutable value object representing a single indexed corpus passage."""

    document_id: str = Field(
        ...,
        description="Stable identifier for the source document (e.g. Wikipedia article title).",
    )
    span_id: str = Field(
        ...,
        description="Identifier for the specific chunk or span within the document.",
    )
    text: str = Field(
        ...,
        description="Raw passage text.",
    )

    model_config = ConfigDict(frozen=True)


class BM25RetrievalDefinition(RetrievalDefinition):
    """Immutable configuration for a BM25 retrieval invocation."""

    top_k: int = Field(
        ...,
        gt=0,
        description="Maximum number of passages to return.",
    )
    score_threshold: float | None = Field(
        default=None,
        description="Optional minimum BM25 score filter.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class FAISSRetrievalDefinition(RetrievalDefinition):
    """Immutable configuration for a FAISS retrieval invocation."""

    top_k: int = Field(
        ...,
        gt=0,
        description="Maximum number of passages to return.",
    )
    similarity_threshold: float | None = Field(
        default=None,
        description="Optional minimum cosine similarity filter.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class HybridRetrievalDefinition(RetrievalDefinition):
    """Immutable configuration for a hybrid retrieval invocation."""

    bm25_top_k: int = Field(
        ...,
        gt=0,
        description="Candidate pool size requested from BM25 constituent.",
    )
    faiss_top_k: int = Field(
        ...,
        gt=0,
        description="Candidate pool size requested from FAISS constituent.",
    )
    top_k: int = Field(
        ...,
        gt=0,
        description="Maximum fused passages to return.",
    )
    rrf_k: int = Field(
        default=60,
        gt=0,
        description="RRF smoothing constant.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class RetrievalMetadata(BaseModel):
    """Minimal immutable execution provenance attached to each EvidenceBundle."""

    strategy_id: str = Field(
        ...,
        description="Identifies which retriever produced this bundle (e.g. 'bm25', 'faiss', 'hybrid').",
    )
    top_k: int = Field(
        ...,
        description="Number of passages requested during retrieval.",
    )

    model_config = ConfigDict(frozen=True)


class EvidencePassage(BaseModel):
    """Immutable representation of a single retrieved evidence unit."""

    document_id: str = Field(
        ...,
        description="Stable identifier for the source document (e.g. Wikipedia article title).",
    )
    span_id: str = Field(
        ...,
        description="Identifier for the specific chunk or span within the document.",
    )
    text: str = Field(
        ...,
        description="Raw passage text.",
    )
    score: float = Field(
        ...,
        description="Relevance score assigned by the retrieval strategy.",
    )
    metadata: Mapping[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional corpus-specific metadata (extensible, JSON-compatible).",
    )

    model_config = ConfigDict(frozen=True)


class EvidenceBundle(BaseModel):
    """Immutable, ordered collection of retrieved passages for a single claim invocation."""

    claim: str = Field(
        ...,
        description="The normalized, verified textual assertion.",
    )
    passages: tuple[EvidencePassage, ...] = Field(
        ...,
        description="Ordered sequence of retrieved passages by descending relevance score.",
    )
    metadata: RetrievalMetadata = Field(
        ...,
        description="Minimal execution provenance for downstream observability.",
    )

    model_config = ConfigDict(frozen=True)
