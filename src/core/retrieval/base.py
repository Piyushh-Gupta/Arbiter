"""Stateless base retriever protocol."""

from typing import Protocol, runtime_checkable

import numpy as np

from src.core.retrieval.retrieval_models import (
    EvidenceBundle,
    RetrievalCandidate,
    RetrievalCandidateSet,
    RetrievalDefinition,
)


@runtime_checkable
class BaseRetriever(Protocol):
    """Stateless protocol for all retrieval strategies."""

    def validate_compatibility(self, definition: RetrievalDefinition) -> None:
        """Statically verifies if this retriever supports the given definition."""
        ...

    def retrieve(self, claim: str, definition: RetrievalDefinition) -> EvidenceBundle:
        """
        Executes the retrieval logic.

        Receives:
        - claim: The verified textual assertion.
        - definition: The validated immutable configuration parameters.

        Returns:
        - EvidenceBundle: A fully materialized, immutable ordered collection of retrieved passages.
        """
        ...


@runtime_checkable
class BaseEncoder(Protocol):
    """Stateless protocol representing shared immutable encoder capabilities."""

    @property
    def model_id(self) -> str:
        """The identifier of the underlying model."""
        ...

    @property
    def embedding_dimension(self) -> int:
        """The size of the output embeddings."""
        ...

    @property
    def device(self) -> str:
        """The execution device (e.g. 'cpu', 'cuda')."""
        ...

    def is_ready(self) -> bool:
        """Indicates if the encoder has loaded its weights and is ready for inference."""
        ...


@runtime_checkable
class QueryEncoder(BaseEncoder, Protocol):
    """Stateless protocol for online encoding of textual queries into dense embeddings."""

    def encode(self, text: str) -> np.ndarray:
        """
        Encodes a textual query into a dense numpy array compatible with the injected vector store.
        The encoder is fully responsible for any required normalization (e.g. L2) before returning.
        """
        ...


@runtime_checkable
class DocumentEncoder(BaseEncoder, Protocol):
    """Stateless protocol for offline batch encoding of corpus documents."""

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encodes a batch of documents for offline indexing.
        """
        ...


@runtime_checkable
class BaseVectorStore(Protocol):
    """Stateless protocol for vector database interactions."""

    def search(self, query: np.ndarray, top_k: int) -> tuple[RetrievalCandidate, ...]:
        """
        Executes a vector similarity search.
        Returns a tuple of RetrievalCandidate instances.
        """
        ...


@runtime_checkable
class BaseCandidateGenerator(Protocol):
    """
    Stateless protocol for generating retrieval candidates.
    Independent of reranking or execution orchestration.
    """

    def generate_candidates(
        self, claim: str, definition: RetrievalDefinition
    ) -> RetrievalCandidateSet:
        """
        Generates an initial set of high-recall evidence candidates.
        """
        ...


@runtime_checkable
class BaseReranker(Protocol):
    """
    Stateless protocol for cross-encoder/late-interaction reranking.
    Independent of vector stores and initial candidate generation.
    """

    def rerank(
        self,
        claim: str,
        candidates: RetrievalCandidateSet,
        definition: RetrievalDefinition,
    ) -> RetrievalCandidateSet:
        """
        Scores and reorders a set of candidates based on the original claim.
        Returns a new RetrievalCandidateSet with normalized scores.
        """
        ...


@runtime_checkable
class IndexBuilder(Protocol):
    """
    Stateful protocol for offline index creation and management.
    Never invoked by the online API.
    """

    def build_index(self, corpus_path: str, index_output_path: str) -> None:
        """
        Ingests a corpus, chunks, encodes, and builds a vector index and metadata store.
        """
        ...
