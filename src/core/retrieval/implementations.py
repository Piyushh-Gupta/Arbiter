"""Concrete implementations of retrieval strategies."""

from collections.abc import Callable, Sequence

import numpy as np
from pydantic import ValidationError
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from src.core.exceptions import RetrievalConfigurationError, RetrievalExecutionError
from src.core.retrieval.base import BaseRetriever
from src.core.retrieval.retrieval_models import (
    BM25RetrievalDefinition,
    CorpusEntry,
    EvidenceBundle,
    EvidencePassage,
    RetrievalDefinition,
    RetrievalMetadata,
)


class BM25Retriever(BaseRetriever):
    """
    Stateless concrete execution strategy for BM25 lexical retrieval.
    """

    def __init__(
        self,
        index: BM25Okapi,
        corpus: Sequence[CorpusEntry],
        tokenizer: Callable[[str], list[str]],
    ) -> None:
        """
        Initializes the retriever with immutable dependencies.
        """
        self._index = index
        self._corpus = tuple(corpus)
        self._tokenizer = tokenizer

    def validate_compatibility(self, definition: RetrievalDefinition) -> None:
        """Fails fast if the definition is not a BM25RetrievalDefinition."""
        if not isinstance(definition, BM25RetrievalDefinition):
            raise RetrievalConfigurationError(
                f"BM25Retriever requires BM25RetrievalDefinition, got {type(definition).__name__}"
            )

    def retrieve(self, claim: str, definition: RetrievalDefinition) -> EvidenceBundle:
        """
        Executes the BM25 retrieval process.
        """
        if not isinstance(definition, BM25RetrievalDefinition):
            raise RetrievalConfigurationError(
                f"BM25Retriever requires BM25RetrievalDefinition, got {type(definition).__name__}"
            )

        try:
            # Tokenize the claim using the injected tokenizer
            tokenized_query = self._tokenizer(claim)

            # Compute scores
            scores = self._index.get_scores(tokenized_query)

            # Find the top-k indices
            # argsort returns indices that would sort the array ascending
            # We negate scores to sort descending in a stable manner
            top_k_indices = np.argsort(-scores)[: definition.top_k]

            passages = []
            for idx in top_k_indices:
                score = float(scores[idx])

                # Apply score threshold if defined
                if (
                    definition.score_threshold is not None
                    and score < definition.score_threshold
                ):
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
                        strategy_id="bm25",
                        top_k=definition.top_k,
                    ),
                )
            except ValidationError as e:
                raise RetrievalExecutionError(
                    f"Failed to construct EvidenceBundle: {e}"
                ) from e

            return bundle

        except Exception as e:
            # Wrap any unhandled unexpected exceptions from rank_bm25 or numpy
            if isinstance(e, RetrievalExecutionError):
                raise
            raise RetrievalExecutionError(
                f"BM25 retrieval execution failed: {e}"
            ) from e
