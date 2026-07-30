"""Concrete implementations of retrieval strategies."""

from collections.abc import Sequence

import faiss
from pydantic import ValidationError

from src.core.exceptions import RetrievalConfigurationError, RetrievalExecutionError
from src.core.retrieval.base import BaseRetriever, QueryEncoder
from src.core.retrieval.hybrid import HybridRetriever
from src.core.retrieval.retrieval_models import (
    CorpusEntry,
    DenseRetrievalDefinition,
    EvidenceBundle,
    EvidencePassage,
    RetrievalDefinition,
    RetrievalMetadata,
)


class DenseRetriever(BaseRetriever):
    """
    Stateless concrete execution strategy for FAISS semantic retrieval.
    """

    def __init__(
        self,
        index: faiss.Index,
        corpus: Sequence[CorpusEntry],
        encoder: QueryEncoder,
    ) -> None:
        """
        Initializes the retriever with immutable dependencies.
        """
        self._index = index
        self._corpus = tuple(corpus)
        self._encoder = encoder

    def validate_compatibility(self, definition: RetrievalDefinition) -> None:
        """Fails fast if the definition is not a DenseRetrievalDefinition."""
        if not isinstance(definition, DenseRetrievalDefinition):
            raise RetrievalConfigurationError(
                f"DenseRetriever requires DenseRetrievalDefinition, got {type(definition).__name__}"
            )

    def retrieve(self, claim: str, definition: RetrievalDefinition) -> EvidenceBundle:
        """
        Executes the FAISS retrieval process.
        """
        if not isinstance(definition, DenseRetrievalDefinition):
            raise RetrievalConfigurationError(
                f"DenseRetriever requires DenseRetrievalDefinition, got {type(definition).__name__}"
            )

        try:
            # Tokenize and encode the claim using the injected encoder
            # The encoder is responsible for normalizing if required by the index contract.
            query_vec = self._encoder.encode(claim)

            # Reshape to (1, d) for FAISS
            query_matrix = query_vec.reshape(1, -1)

            # Search the index
            distances, indices = self._index.search(query_matrix, definition.top_k)

            # Flatten results
            flat_distances = distances.flatten()
            flat_indices = indices.flatten()

            passages = []
            for i in range(len(flat_indices)):
                idx = int(flat_indices[i])

                # FAISS returns -1 when there are fewer results than top_k
                if idx == -1:
                    continue

                score = float(flat_distances[i])

                # Apply score threshold if defined
                if definition.min_score is not None and score < definition.min_score:
                    continue

                entry = self._corpus[idx]

                try:
                    passage = EvidencePassage(
                        document_id=entry.document_id,
                        span_id=entry.span_id,
                        text=entry.text,
                        score=score,
                    )
                    passages.append(passage)
                except ValidationError as e:
                    raise RetrievalExecutionError(
                        f"Failed to construct EvidencePassage for index {idx}: {e}"
                    ) from e

            # Construct bundle
            try:
                bundle = EvidenceBundle(
                    claim=claim,
                    passages=tuple(passages),
                    metadata=RetrievalMetadata(
                        strategy_id="faiss",
                        top_k=definition.top_k,
                    ),
                )
            except ValidationError as e:
                raise RetrievalExecutionError(
                    f"Failed to construct EvidenceBundle: {e}"
                ) from e

            return bundle

        except Exception as e:
            # Wrap any unhandled unexpected exceptions from faiss or numpy
            if isinstance(e, RetrievalExecutionError):
                raise
            raise RetrievalExecutionError(
                f"FAISS retrieval execution failed: {e}"
            ) from e


__all__ = ["DenseRetriever", "HybridRetriever"]
