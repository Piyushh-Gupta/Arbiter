"""Stateless base reranking protocols."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from src.core.reranking.reranking_models import RerankingDefinition
from src.core.retrieval.retrieval_models import EvidenceBundle, RetrievalCandidateSet


@runtime_checkable
class BaseCrossEncoderScorer(Protocol):
    """Stateless protocol for cross-encoder model batch scoring."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """
        Scores a sequence of passages against a query string.
        Returns a list of float relevance scores matching the order of input passages.
        """
        ...


# Alias for backward compatibility
CrossEncoderScorer = BaseCrossEncoderScorer


@runtime_checkable
class BaseReranker(Protocol):
    """Stateless protocol for candidate reranking strategies."""

    def validate_compatibility(self, definition: RerankingDefinition) -> None:
        """Fails fast if the definition is incompatible with the reranker strategy."""
        ...

    def rerank(
        self,
        claim: str,
        candidates: RetrievalCandidateSet | EvidenceBundle,
        definition: RerankingDefinition,
    ) -> EvidenceBundle:
        """
        Scores and reorders candidates using the original claim and materializes an EvidenceBundle.
        """
        ...
