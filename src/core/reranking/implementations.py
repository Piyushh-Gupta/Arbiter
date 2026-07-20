"""Concrete implementations of reranking strategies."""

from pydantic import ValidationError

from src.core.exceptions import RerankingConfigurationError, RerankingExecutionError
from src.core.reranking.base import CrossEncoderScorer
from src.core.reranking.reranking_models import RerankingDefinition
from src.core.retrieval.retrieval_models import (
    EvidenceBundle,
    EvidencePassage,
    RetrievalMetadata,
)


class CrossEncoderReranker:
    """
    Stateless concrete execution strategy for cross-encoder reranking.
    """

    def __init__(
        self,
        scorer: CrossEncoderScorer,
    ) -> None:
        """
        Initializes the reranker with an immutable scorer dependency.
        """
        self._scorer = scorer

    def validate_compatibility(self, definition: RerankingDefinition) -> None:
        """Fails fast if the definition is not a RerankingDefinition."""
        if not isinstance(definition, RerankingDefinition):
            raise RerankingConfigurationError(
                f"CrossEncoderReranker requires RerankingDefinition, got {type(definition).__name__}"
            )

    def rerank(
        self, bundle: EvidenceBundle, definition: RerankingDefinition
    ) -> EvidenceBundle:
        """
        Executes cross-encoder reranking on the given bundle.
        """
        self.validate_compatibility(definition)

        if not bundle.passages:
            try:
                return EvidenceBundle(
                    claim=bundle.claim,
                    passages=(),
                    metadata=RetrievalMetadata(
                        strategy_id="cross_encoder",
                        top_k=definition.top_k,
                    ),
                )
            except ValidationError as e:
                raise RerankingExecutionError(
                    f"Failed to construct empty reranked bundle: {e}"
                ) from e

        passage_texts = [p.text for p in bundle.passages]
        try:
            scores = self._scorer.score(bundle.claim, passage_texts)
        except Exception as e:
            if isinstance(e, RerankingExecutionError):
                raise
            raise RerankingExecutionError(f"Cross-encoder scoring failed: {e}") from e

        if len(scores) != len(bundle.passages):
            raise RerankingExecutionError(
                f"Cross-encoder scorer returned {len(scores)} scores for {len(bundle.passages)} passages."
            )

        pairs = list(zip(bundle.passages, scores, strict=True))

        # Sort by descending cross-encoder score.
        # Break ties lexicographically by (document_id, span_id)
        sorted_pairs = sorted(
            pairs, key=lambda p: (-p[1], p[0].document_id, p[0].span_id)
        )

        if definition.score_threshold is not None:
            sorted_pairs = [
                p for p in sorted_pairs if p[1] >= definition.score_threshold
            ]

        top_pairs = sorted_pairs[: definition.top_k]
        reranked_passages = []

        for original_passage, new_score in top_pairs:
            merged_metadata = {
                **original_passage.metadata,
                "retrieval_score": original_passage.score,
            }
            try:
                new_passage = EvidencePassage(
                    document_id=original_passage.document_id,
                    span_id=original_passage.span_id,
                    text=original_passage.text,
                    score=new_score,
                    metadata=merged_metadata,
                )
                reranked_passages.append(new_passage)
            except ValidationError as e:
                raise RerankingExecutionError(
                    f"Failed to construct reranked EvidencePassage: {e}"
                ) from e

        try:
            new_bundle = EvidenceBundle(
                claim=bundle.claim,
                passages=tuple(reranked_passages),
                metadata=RetrievalMetadata(
                    strategy_id="cross_encoder", top_k=definition.top_k
                ),
            )
        except ValidationError as e:
            raise RerankingExecutionError(
                f"Failed to construct reranked EvidenceBundle: {e}"
            ) from e

        return new_bundle
