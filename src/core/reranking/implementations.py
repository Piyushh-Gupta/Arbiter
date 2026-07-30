"""Concrete implementations of reranking strategies."""

from collections.abc import Sequence
from typing import Any

from src.core.exceptions import RerankingConfigurationError, RerankingExecutionError
from src.core.reranking.base import BaseCrossEncoderScorer, BaseReranker
from src.core.reranking.reranking_models import (
    CrossEncoderModelMetadata,
    RerankingDefinition,
)
from src.core.retrieval.base import DocumentStore
from src.core.retrieval.retrieval_models import (
    EvidenceBundle,
    EvidencePassage,
    RerankMetadata,
    RetrievalCandidateSet,
    RetrievalMetadata,
)


def _extract_corpus_index(metadata: Any) -> int:
    if isinstance(metadata, dict):
        val = metadata.get("corpus_index", 0)
        if isinstance(val, (int, float, str)):
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0
    return 0


class SentenceTransformerCrossEncoderScorer(BaseCrossEncoderScorer):
    """
    Stateless scorer wrapper wrapping sentence-transformers CrossEncoder.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        model_metadata: CrossEncoderModelMetadata | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self._model_id = model_id
        self._device = device
        try:
            self._model = CrossEncoder(model_name=self._model_id, device=self._device)
        except Exception as e:
            raise RerankingConfigurationError(
                f"Failed to load CrossEncoder model '{model_id}' on device '{device}': {e}"
            ) from e

        if model_metadata is not None:
            self._metadata = model_metadata
        else:
            max_seq_len = getattr(self._model, "max_seq_length", 512) or 512
            self._metadata = CrossEncoderModelMetadata(
                model_identifier=self._model_id,
                tokenizer_identifier=self._model_id,
                inference_framework="sentence-transformers",
                execution_device=self._device,
                max_sequence_length=max_seq_len,
            )

    @property
    def metadata(self) -> CrossEncoderModelMetadata:
        return self._metadata

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        try:
            pairs = [[query, passage] for passage in passages]
            scores = self._model.predict(pairs, show_progress_bar=False)
            return [float(s) for s in scores]
        except Exception as e:
            raise RerankingExecutionError(
                f"Cross-encoder batch inference failed: {e}"
            ) from e


class CrossEncoderReranker(BaseReranker):
    """
    Stateless concrete execution strategy for cross-encoder candidate reranking.
    """

    def __init__(
        self,
        scorer: BaseCrossEncoderScorer,
        document_store: DocumentStore | None = None,
    ) -> None:
        """
        Initializes CrossEncoderReranker with an immutable scorer and optional DocumentStore.
        """
        self._scorer = scorer
        self._document_store = document_store

    def validate_compatibility(self, definition: RerankingDefinition) -> None:
        """Fails fast if the definition is not a RerankingDefinition."""
        if not isinstance(definition, RerankingDefinition):
            raise RerankingConfigurationError(
                f"CrossEncoderReranker requires RerankingDefinition, got {type(definition).__name__}"
            )

    def rerank(
        self,
        claim: str | EvidenceBundle,
        candidates: RetrievalCandidateSet
        | EvidenceBundle
        | RerankingDefinition
        | None = None,
        definition: RerankingDefinition | None = None,
    ) -> EvidenceBundle:
        """
        Executes cross-encoder reranking over input candidates or bundle.
        Materializes chunk text strictly for the top_k_input candidates via DocumentStore.
        """
        # Handle overloaded signature (claim: EvidenceBundle, definition: RerankingDefinition)
        if isinstance(claim, EvidenceBundle) and isinstance(
            candidates, RerankingDefinition
        ):
            bundle = claim
            def_obj = candidates
            self.validate_compatibility(def_obj)

            if not bundle.passages:
                return EvidenceBundle(
                    claim=bundle.claim,
                    passages=(),
                    metadata=RetrievalMetadata(
                        strategy_id="cross_encoder", top_k=def_obj.top_k_output
                    ),
                )

            top_input_passages = bundle.passages[: def_obj.top_k_input]
            passage_texts = [p.text for p in top_input_passages]

            scores: list[float] = []
            try:
                for i in range(0, len(passage_texts), def_obj.batch_size):
                    batch = passage_texts[i : i + def_obj.batch_size]
                    batch_scores = self._scorer.score(bundle.claim, batch)
                    scores.extend(batch_scores)
            except Exception as e:
                if isinstance(
                    e, (RerankingExecutionError, RerankingConfigurationError)
                ):
                    raise
                raise RerankingExecutionError(
                    f"Cross-encoder batch inference failed: {e}"
                ) from e

            if len(scores) != len(top_input_passages):
                raise RerankingExecutionError(
                    f"Cross-encoder scorer returned {len(scores)} scores for {len(top_input_passages)} passages."
                )

            scored_pairs = []
            for p, s in zip(top_input_passages, scores):
                if def_obj.score_threshold is not None and s < def_obj.score_threshold:
                    continue
                corpus_idx = _extract_corpus_index(p.metadata)
                scored_pairs.append((p, s, corpus_idx))

            scored_pairs.sort(
                key=lambda item: (
                    -item[1],
                    item[2],
                    item[0].document_id,
                    item[0].span_id,
                )
            )
            top_pairs = scored_pairs[: def_obj.top_k_output]

            reranked_passages = []
            for rank_idx, (orig_p, score, corpus_idx) in enumerate(top_pairs, start=1):
                rerank_meta = RerankMetadata(
                    rerank_score=score,
                    rerank_rank=rank_idx,
                    prior_fusion_metadata=orig_p.fusion_metadata,
                    reranking_model_id=def_obj.model_identifier,
                )
                merged_metadata = dict(orig_p.metadata)
                merged_metadata["retrieval_score"] = orig_p.score

                new_p = EvidencePassage(
                    document_id=orig_p.document_id,
                    span_id=orig_p.span_id,
                    text=orig_p.text,
                    score=score,
                    metadata=merged_metadata,
                    fusion_metadata=orig_p.fusion_metadata,
                    rerank_metadata=rerank_meta,
                )
                reranked_passages.append(new_p)

            return EvidenceBundle(
                claim=bundle.claim,
                passages=tuple(reranked_passages),
                metadata=RetrievalMetadata(
                    strategy_id="cross_encoder", top_k=def_obj.top_k_output
                ),
            )

        # Standard signature (claim: str, candidates: RetrievalCandidateSet|EvidenceBundle, definition: RerankingDefinition)
        claim_str = str(claim)
        if definition is None:
            raise RerankingConfigurationError("RerankingDefinition required")

        self.validate_compatibility(definition)

        if isinstance(candidates, EvidenceBundle):
            return self.rerank(candidates, definition)

        if not isinstance(candidates, RetrievalCandidateSet):
            raise RerankingConfigurationError(
                "Expected RetrievalCandidateSet or EvidenceBundle"
            )

        if self._document_store is None:
            raise RerankingConfigurationError(
                "CrossEncoderReranker requires a DocumentStore to resolve candidate text."
            )

        input_candidates = candidates.candidates[: definition.top_k_input]
        if not input_candidates:
            return EvidenceBundle(
                claim=claim_str,
                passages=(),
                metadata=RetrievalMetadata(
                    strategy_id="cross_encoder", top_k=definition.top_k_output
                ),
            )

        chunks = [self._document_store.get_chunk(c.span_id) for c in input_candidates]
        passage_texts = [c.text for c in chunks]

        scores = []
        try:
            for i in range(0, len(passage_texts), definition.batch_size):
                batch = passage_texts[i : i + definition.batch_size]
                batch_scores = self._scorer.score(claim_str, batch)
                scores.extend(batch_scores)
        except Exception as e:
            if isinstance(e, (RerankingExecutionError, RerankingConfigurationError)):
                raise
            raise RerankingExecutionError(
                f"Cross-encoder batch inference failed: {e}"
            ) from e

        if len(scores) != len(input_candidates):
            raise RerankingExecutionError(
                f"Cross-encoder scorer returned {len(scores)} scores for {len(input_candidates)} passages."
            )

        scored_triples = []
        for cand, chunk, score in zip(input_candidates, chunks, scores):
            if (
                definition.score_threshold is not None
                and score < definition.score_threshold
            ):
                continue
            corpus_idx = _extract_corpus_index(cand.metadata)
            scored_triples.append((cand, chunk, score, corpus_idx))

        scored_triples.sort(
            key=lambda item: (-item[2], item[3], item[1].document_id, item[1].span_id)
        )
        top_triples = scored_triples[: definition.top_k_output]

        reranked_passages = []
        for rank_idx, (cand, chunk, score, corpus_idx) in enumerate(
            top_triples, start=1
        ):
            rerank_meta = RerankMetadata(
                rerank_score=score,
                rerank_rank=rank_idx,
                prior_fusion_metadata=cand.fusion_metadata,
                reranking_model_id=definition.model_identifier,
            )
            merged_metadata = dict(chunk.metadata)
            merged_metadata["retrieval_score"] = cand.score

            passage = EvidencePassage(
                document_id=chunk.document_id,
                span_id=chunk.span_id,
                text=chunk.text,
                score=score,
                metadata=merged_metadata,
                fusion_metadata=cand.fusion_metadata,
                rerank_metadata=rerank_meta,
            )
            reranked_passages.append(passage)

        return EvidenceBundle(
            claim=claim_str,
            passages=tuple(reranked_passages),
            metadata=RetrievalMetadata(
                strategy_id="cross_encoder", top_k=definition.top_k_output
            ),
        )
