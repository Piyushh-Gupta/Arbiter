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
class QueryEncoder(Protocol):
    """Stateless protocol for encoding textual queries into dense embeddings."""

    def encode(self, text: str) -> np.ndarray:
        """
        Encodes a textual query into a dense numpy array compatible with the injected FAISS index.
        The encoder is fully responsible for any required normalization (e.g. L2) before returning.
        """
        ...
