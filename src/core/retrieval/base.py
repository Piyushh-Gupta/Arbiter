"""Stateless base retriever protocol."""

from typing import Protocol, runtime_checkable

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
