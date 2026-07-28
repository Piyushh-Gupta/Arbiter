"""Stateless base retriever protocol."""

from typing import Protocol, runtime_checkable

import numpy as np

from src.core.retrieval.retrieval_models import EvidenceBundle, RetrievalDefinition


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
    """Stateless protocol for encoding textual inputs into dense embeddings."""
    pass


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
    
    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Executes a vector similarity search.
        Returns a tuple of (distances, indices).
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
    ) -> EvidenceBundle:
        """
        Generates an initial set of evidence candidates based on the claim.
        """
        ...
