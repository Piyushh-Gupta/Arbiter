"""Stateless concrete strategy for Hybrid Retrieval using Reciprocal Rank Fusion (RRF)."""

from typing import Any

from src.core.exceptions import RetrievalConfigurationError, RetrievalExecutionError
from src.core.retrieval.base import BaseCandidateGenerator, BaseRetriever, DocumentStore
from src.core.retrieval.retrieval_models import (
    EvidenceBundle,
    EvidencePassage,
    FusionMetadata,
    HybridRetrievalDefinition,
    RetrievalCandidate,
    RetrievalCandidateSet,
    RetrievalDefinition,
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


class HybridRetriever(BaseRetriever):
    """
    Stateless concrete execution strategy for Hybrid (BM25 + Dense) retrieval orchestration using RRF.
    """

    def __init__(
        self,
        bm25_generator: BaseCandidateGenerator | None = None,
        dense_generator: BaseCandidateGenerator | None = None,
        document_store: DocumentStore | None = None,
    ) -> None:
        """
        Initializes HybridRetriever with immutable candidate generators and document store.
        At least one generator and a document store must be provided.
        """
        if bm25_generator is None and dense_generator is None:
            raise ValueError(
                "HybridRetriever requires at least one candidate generator (bm25_generator or dense_generator)."
            )
        if document_store is None:
            raise ValueError("HybridRetriever requires a DocumentStore instance.")

        self._bm25_generator = bm25_generator
        self._dense_generator = dense_generator
        self._document_store = document_store

    def validate_compatibility(self, definition: RetrievalDefinition) -> None:
        """Fails fast if definition is incompatible or missing required candidate generators."""
        if not isinstance(definition, HybridRetrievalDefinition):
            raise RetrievalConfigurationError(
                f"HybridRetriever requires HybridRetrievalDefinition, got {type(definition).__name__}"
            )

        if definition.bm25_definition is not None and self._bm25_generator is None:
            raise RetrievalConfigurationError(
                "HybridRetrievalDefinition specified bm25_definition, but no BM25CandidateGenerator was provided to HybridRetriever."
            )

        if definition.dense_definition is not None and self._dense_generator is None:
            raise RetrievalConfigurationError(
                "HybridRetrievalDefinition specified dense_definition, but no DenseCandidateGenerator was provided to HybridRetriever."
            )

    def fuse_candidate_sets(
        self,
        lexical_candidates: RetrievalCandidateSet | None,
        dense_candidates: RetrievalCandidateSet | None,
        rrf_k: int,
        top_k: int,
    ) -> RetrievalCandidateSet:
        """
        Statelessly merges candidate sets using Reciprocal Rank Fusion (RRF) and deterministic corpus-order tie breaking.
        """
        # Map span_id -> accumulator data
        # {span_id: {"lex_rank": int|None, "lex_score": float|None, "dense_rank": int|None, "dense_score": float|None, "corpus_index": int}}
        candidate_map: dict[str, dict[str, Any]] = {}

        if lexical_candidates is not None:
            for rank_idx, candidate in enumerate(
                lexical_candidates.candidates, start=1
            ):
                span_id = candidate.span_id
                corpus_index = _extract_corpus_index(candidate.metadata)
                if span_id not in candidate_map:
                    candidate_map[span_id] = {
                        "lex_rank": rank_idx,
                        "lex_score": candidate.score,
                        "dense_rank": None,
                        "dense_score": None,
                        "corpus_index": corpus_index,
                        "sources": ["bm25"],
                    }
                else:
                    candidate_map[span_id]["lex_rank"] = rank_idx
                    candidate_map[span_id]["lex_score"] = candidate.score
                    if "bm25" not in candidate_map[span_id]["sources"]:
                        candidate_map[span_id]["sources"].append("bm25")

        if dense_candidates is not None:
            for rank_idx, candidate in enumerate(dense_candidates.candidates, start=1):
                span_id = candidate.span_id
                corpus_index = _extract_corpus_index(candidate.metadata)
                if span_id not in candidate_map:
                    candidate_map[span_id] = {
                        "lex_rank": None,
                        "lex_score": None,
                        "dense_rank": rank_idx,
                        "dense_score": candidate.score,
                        "corpus_index": corpus_index,
                        "sources": ["dense"],
                    }
                else:
                    candidate_map[span_id]["dense_rank"] = rank_idx
                    candidate_map[span_id]["dense_score"] = candidate.score
                    if "dense" not in candidate_map[span_id]["sources"]:
                        candidate_map[span_id]["sources"].append("dense")

        fused_candidates: list[RetrievalCandidate] = []
        for span_id, data in candidate_map.items():
            rrf_score = 0.0
            if data["lex_rank"] is not None:
                rrf_score += 1.0 / (rrf_k + data["lex_rank"])
            if data["dense_rank"] is not None:
                rrf_score += 1.0 / (rrf_k + data["dense_rank"])

            fusion_meta = FusionMetadata(
                lexical_rank=data["lex_rank"],
                lexical_score=data["lex_score"],
                semantic_rank=data["dense_rank"],
                semantic_score=data["dense_score"],
                rrf_score=rrf_score,
                retrieval_sources=tuple(data["sources"]),
            )

            fused_candidates.append(
                RetrievalCandidate(
                    span_id=span_id,
                    score=rrf_score,
                    metadata={"corpus_index": data["corpus_index"]},
                    fusion_metadata=fusion_meta,
                )
            )

        # Deterministic sorting: 1) descending RRF score, 2) ascending corpus_index (original corpus insertion order)
        fused_candidates.sort(
            key=lambda c: (-c.score, _extract_corpus_index(c.metadata))
        )

        truncated = tuple(fused_candidates[:top_k])

        return RetrievalCandidateSet(
            candidates=truncated,
            metadata=RetrievalMetadata(strategy_id="hybrid", top_k=top_k),
        )

    def retrieve(self, claim: str, definition: RetrievalDefinition) -> EvidenceBundle:
        """
        Executes hybrid retrieval by retrieving candidates from constituents, fusing via RRF,
        resolving chunks from DocumentStore, and materializing an EvidenceBundle.
        """
        self.validate_compatibility(definition)

        if not isinstance(definition, HybridRetrievalDefinition):
            # Narrowing for type checker
            raise RetrievalConfigurationError(
                f"Expected HybridRetrievalDefinition, got {type(definition).__name__}"
            )

        try:
            lexical_candidates: RetrievalCandidateSet | None = None
            if (
                definition.bm25_definition is not None
                and self._bm25_generator is not None
            ):
                lexical_candidates = self._bm25_generator.generate_candidates(
                    claim, definition.bm25_definition
                )

            dense_candidates: RetrievalCandidateSet | None = None
            if (
                definition.dense_definition is not None
                and self._dense_generator is not None
            ):
                dense_candidates = self._dense_generator.generate_candidates(
                    claim, definition.dense_definition
                )

            candidate_set = self.fuse_candidate_sets(
                lexical_candidates=lexical_candidates,
                dense_candidates=dense_candidates,
                rrf_k=definition.rrf_k,
                top_k=definition.top_k,
            )

            # Resolve chunks through DocumentStore exclusively for the top_k fused candidates
            passages: list[EvidencePassage] = []
            for candidate in candidate_set.candidates:
                chunk = self._document_store.get_chunk(candidate.span_id)
                passages.append(
                    EvidencePassage(
                        document_id=chunk.document_id,
                        span_id=chunk.span_id,
                        text=chunk.text,
                        score=candidate.score,
                        metadata=chunk.metadata,
                        fusion_metadata=candidate.fusion_metadata,
                    )
                )

            return EvidenceBundle(
                claim=claim,
                passages=tuple(passages),
                metadata=RetrievalMetadata(
                    strategy_id="hybrid",
                    top_k=definition.top_k,
                ),
            )
        except Exception as e:
            if isinstance(e, (RetrievalExecutionError, RetrievalConfigurationError)):
                raise
            raise RetrievalExecutionError(
                f"Hybrid retrieval execution failed: {e}"
            ) from e
