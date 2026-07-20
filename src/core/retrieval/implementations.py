"""Concrete implementations of retrieval strategies."""

from collections.abc import Callable, Sequence

import faiss
import numpy as np
from pydantic import ValidationError
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from src.core.exceptions import RetrievalConfigurationError, RetrievalExecutionError
from src.core.retrieval.base import BaseRetriever, QueryEncoder
from src.core.retrieval.retrieval_models import (
    BM25RetrievalDefinition,
    CorpusEntry,
    EvidenceBundle,
    EvidencePassage,
    FAISSRetrievalDefinition,
    HybridRetrievalDefinition,
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


class FAISSRetriever(BaseRetriever):
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
        """Fails fast if the definition is not a FAISSRetrievalDefinition."""
        if not isinstance(definition, FAISSRetrievalDefinition):
            raise RetrievalConfigurationError(
                f"FAISSRetriever requires FAISSRetrievalDefinition, got {type(definition).__name__}"
            )

    def retrieve(self, claim: str, definition: RetrievalDefinition) -> EvidenceBundle:
        """
        Executes the FAISS retrieval process.
        """
        if not isinstance(definition, FAISSRetrievalDefinition):
            raise RetrievalConfigurationError(
                f"FAISSRetriever requires FAISSRetrievalDefinition, got {type(definition).__name__}"
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
                if (
                    definition.similarity_threshold is not None
                    and score < definition.similarity_threshold
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


class HybridRetriever(BaseRetriever):
    """
    Stateless concrete execution strategy for hybrid (BM25 + FAISS) retrieval
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        faiss_retriever: FAISSRetriever,
    ) -> None:
        """
        Initializes the hybrid retriever with fully constructed constituents.
        """
        self._bm25 = bm25_retriever
        self._faiss = faiss_retriever

    def validate_compatibility(self, definition: RetrievalDefinition) -> None:
        """Fails fast if the definition is not a HybridRetrievalDefinition."""
        if not isinstance(definition, HybridRetrievalDefinition):
            raise RetrievalConfigurationError(
                f"HybridRetriever requires HybridRetrievalDefinition, got {type(definition).__name__}"
            )

    def retrieve(self, claim: str, definition: RetrievalDefinition) -> EvidenceBundle:
        """
        Executes hybrid retrieval and fuses results via RRF.
        """
        if not isinstance(definition, HybridRetrievalDefinition):
            raise RetrievalConfigurationError(
                f"HybridRetriever requires HybridRetrievalDefinition, got {type(definition).__name__}"
            )

        try:
            # 1. Execute constituent retrievers with ephemeral threshold-free definitions
            bm25_def = BM25RetrievalDefinition(top_k=definition.bm25_top_k)
            faiss_def = FAISSRetrievalDefinition(top_k=definition.faiss_top_k)

            bm25_bundle = self._bm25.retrieve(claim, bm25_def)
            faiss_bundle = self._faiss.retrieve(claim, faiss_def)

            # 2. Compute RRF scores
            # Use (document_id, span_id) as the identity key.
            rrf_scores: dict[tuple[str, str], float] = {}
            passage_map: dict[tuple[str, str], EvidencePassage] = {}

            # BM25 ranks
            for rank_zero_indexed, passage in enumerate(bm25_bundle.passages):
                rank = rank_zero_indexed + 1
                key = (passage.document_id, passage.span_id)
                if key not in passage_map:
                    passage_map[key] = passage
                if key not in rrf_scores:
                    rrf_scores[key] = 0.0
                rrf_scores[key] += 1.0 / (definition.rrf_k + rank)

            # FAISS ranks
            for rank_zero_indexed, passage in enumerate(faiss_bundle.passages):
                rank = rank_zero_indexed + 1
                key = (passage.document_id, passage.span_id)
                if key not in passage_map:
                    passage_map[key] = passage
                if key not in rrf_scores:
                    rrf_scores[key] = 0.0
                rrf_scores[key] += 1.0 / (definition.rrf_k + rank)

            # 3. Sort by descending RRF score, breaking ties lexicographically by (document_id, span_id)
            sorted_keys = sorted(
                rrf_scores.keys(), key=lambda k: (-rrf_scores[k], k[0], k[1])
            )

            # 4. Truncate to top_k and assemble bundle
            top_keys = sorted_keys[: definition.top_k]
            fused_passages = []

            for key in top_keys:
                original_passage = passage_map[key]
                try:
                    fused_passage = EvidencePassage(
                        document_id=original_passage.document_id,
                        span_id=original_passage.span_id,
                        text=original_passage.text,
                        score=rrf_scores[key],
                        metadata=original_passage.metadata,
                    )
                    fused_passages.append(fused_passage)
                except ValidationError as e:
                    raise RetrievalExecutionError(
                        f"Failed to construct EvidencePassage for fused key {key}: {e}"
                    ) from e

            try:
                bundle = EvidenceBundle(
                    claim=claim,
                    passages=tuple(fused_passages),
                    metadata=RetrievalMetadata(
                        strategy_id="hybrid",
                        top_k=definition.top_k,
                    ),
                )
            except ValidationError as e:
                raise RetrievalExecutionError(
                    f"Failed to construct EvidenceBundle: {e}"
                ) from e

            return bundle

        except Exception as e:
            if isinstance(e, RetrievalExecutionError):
                raise
            raise RetrievalExecutionError(
                f"Hybrid retrieval execution failed: {e}"
            ) from e
