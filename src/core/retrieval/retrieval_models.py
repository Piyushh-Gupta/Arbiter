"""Immutable domain models for the Evidence Retrieval subsystem."""

import typing
from typing import Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    model_validator,
)

if typing.TYPE_CHECKING:
    from src.core.retrieval.base import BaseRetriever
else:
    BaseRetriever = typing.Any


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
    min_score: float | None = Field(
        default=None,
        description="Optional minimum BM25 score filter.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class DenseRetrievalDefinition(RetrievalDefinition):
    """Immutable configuration for a FAISS retrieval invocation."""

    top_k: int = Field(
        ...,
        gt=0,
        description="Maximum number of passages to return.",
    )
    min_score: float | None = Field(
        default=None,
        description="Optional minimum cosine similarity filter (min_score).",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class FusionMetadata(BaseModel):
    """Immutable metadata tracking multi-source retrieval provenance and rank metrics."""

    lexical_rank: int | None = Field(
        default=None,
        description="1-based rank in BM25 candidate list, if retrieved lexically.",
    )
    lexical_score: float | None = Field(
        default=None,
        description="Original score from BM25 candidate generator, if retrieved lexically.",
    )
    semantic_rank: int | None = Field(
        default=None,
        description="1-based rank in Dense candidate list, if retrieved semantically.",
    )
    semantic_score: float | None = Field(
        default=None,
        description="Original score from Dense candidate generator, if retrieved semantically.",
    )
    rrf_score: float = Field(
        ...,
        description="Reciprocal Rank Fusion score computed for this candidate.",
    )
    retrieval_sources: tuple[str, ...] = Field(
        ...,
        description="Tuple of contributing strategy IDs (e.g., ('bm25', 'dense')).",
    )

    model_config = ConfigDict(frozen=True)


class RerankMetadata(BaseModel):
    """Immutable metadata recording stage-2 cross-encoder reranking provenance."""

    rerank_score: float = Field(
        ...,
        description="Raw Cross-Encoder relevance score assigned to this candidate.",
    )
    rerank_rank: int = Field(
        ...,
        description="1-based rank assigned after cross-encoder reranking.",
    )
    prior_fusion_metadata: FusionMetadata | None = Field(
        default=None,
        description="Preserved stage-1 hybrid fusion metadata.",
    )
    reranking_model_id: str = Field(
        ...,
        description="Model identifier used for cross-encoder reranking.",
    )

    model_config = ConfigDict(frozen=True)


class HybridRetrievalDefinition(RetrievalDefinition):
    """Immutable configuration for a hybrid retrieval invocation."""

    bm25_definition: BM25RetrievalDefinition | None = Field(
        default=None,
        description="Optional immutable configuration for BM25 lexical candidate generator.",
    )
    dense_definition: DenseRetrievalDefinition | None = Field(
        default=None,
        description="Optional immutable configuration for Dense semantic candidate generator.",
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

    @model_validator(mode="after")
    def _validate_at_least_one_definition(self) -> "HybridRetrievalDefinition":
        if self.bm25_definition is None and self.dense_definition is None:
            raise ValueError(
                "HybridRetrievalDefinition requires at least one constituent definition (bm25_definition or dense_definition)."
            )
        return self


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
    fusion_metadata: FusionMetadata | None = Field(
        default=None,
        description="Structured provenance details if candidate was produced via hybrid retrieval.",
    )
    rerank_metadata: RerankMetadata | None = Field(
        default=None,
        description="Structured provenance details if candidate was reranked by a cross-encoder.",
    )

    model_config = ConfigDict(frozen=True)


class RetrievalCandidate(BaseModel):
    """Minimal immutable structure representing a single high-recall retrieval hit before it is materialized into an EvidencePassage."""

    span_id: str = Field(
        ...,
        description="Identifier for the specific chunk or span within the document.",
    )
    score: float = Field(
        ...,
        description="Raw or normalized score assigned by the retrieval source.",
    )
    metadata: Mapping[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional vector-store metadata (e.g., source document mappings).",
    )
    fusion_metadata: FusionMetadata | None = Field(
        default=None,
        description="Structured provenance details if candidate was produced via hybrid retrieval.",
    )
    rerank_metadata: RerankMetadata | None = Field(
        default=None,
        description="Structured provenance details if candidate was reranked by a cross-encoder.",
    )

    model_config = ConfigDict(frozen=True)


class RetrievalCandidateSet(BaseModel):
    """Immutable, ordered collection of retrieval candidates."""

    candidates: tuple[RetrievalCandidate, ...] = Field(
        ...,
        description="Ordered sequence of retrieved candidates by descending relevance score.",
    )
    metadata: RetrievalMetadata = Field(
        ...,
        description="Minimal execution provenance for downstream observability.",
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


class RetrievalProfile(BaseModel):
    """Immutable reusable wrapper binding a retrieval definition to its execution strategy."""

    profile_id: str = Field(
        ..., description="Unique identifier for this retrieval profile."
    )
    definition: RetrievalDefinition = Field(
        ...,
        description="The strictly immutable configuration for this retrieval strategy.",
    )
    strategy: "BaseRetriever" = Field(
        ..., description="The stateless executable strategy resolving the definition."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "RetrievalProfile":
        """Statically verifies compatibility between the definition and strategy upon construction."""
        self.strategy.validate_compatibility(self.definition)
        return self


class RetrievalProfileRegistry(BaseModel):
    """Immutable namespace for securely resolving named retrieval profiles."""

    profiles: tuple[RetrievalProfile, ...] = Field(
        ...,
        min_length=1,
        description="The abstract collection of registered retrieval profiles.",
    )

    _profile_index: dict[str, RetrievalProfile] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "RetrievalProfileRegistry":
        from src.core.exceptions import DuplicateRetrievalProfileError

        index: dict[str, RetrievalProfile] = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicateRetrievalProfileError(
                    f"Duplicate retrieval profile identifier: {profile.profile_id}"
                )
            index[profile.profile_id] = profile

        # Bypass Pydantic's frozen constraint to initialize the O(1) private lookup table
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> RetrievalProfile:
        """Resolves a profile statelessly in O(1) time."""
        from src.core.exceptions import RetrievalProfileNotFoundError

        if profile_id not in self._profile_index:
            raise RetrievalProfileNotFoundError(
                f"Retrieval profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]
