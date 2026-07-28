from collections.abc import Sequence
from pathlib import Path

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from src.core.exceptions import RetrievalConfigurationError, RetrievalExecutionError
from src.core.indexing.models import Chunk
from src.core.retrieval.base import (
    BaseCandidateGenerator,
    BaseRetriever,
    DocumentStore,
    Tokenizer,
)
from src.core.retrieval.retrieval_models import (
    BM25RetrievalDefinition,
    EvidenceBundle,
    EvidencePassage,
    RetrievalCandidate,
    RetrievalCandidateSet,
    RetrievalDefinition,
    RetrievalMetadata,
)


class WhitespaceTokenizer(Tokenizer):
    """Simple deterministic whitespace tokenizer."""

    def tokenize(self, text: str) -> list[str]:
        return text.lower().split()


class MetadataDocumentStore(DocumentStore):
    """
    Document store backed by the metadata.jsonl generated during offline indexing.
    Loads all chunks into memory.
    """

    def __init__(self, metadata_path: str | Path) -> None:
        self._chunks: dict[str, Chunk] = {}
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = Chunk.model_validate_json(line)
                    self._chunks[chunk.span_id] = chunk

    def get_chunk(self, span_id: str) -> Chunk:
        if span_id not in self._chunks:
            raise RetrievalExecutionError(f"Chunk with span_id {span_id} not found in store.")
        return self._chunks[span_id]


class BM25CandidateGenerator(BaseCandidateGenerator):
    """
    Stateless candidate generator for BM25 lexical retrieval.
    """

    def __init__(
        self,
        index: BM25Okapi,
        span_ids: Sequence[str],
        tokenizer: Tokenizer,
    ) -> None:
        """
        Initializes the candidate generator with immutable dependencies.
        span_ids must exactly match the corpus order used to build the BM25Okapi index.
        """
        self._index = index
        self._span_ids = tuple(span_ids)
        self._tokenizer = tokenizer

        if len(self._span_ids) != self._index.corpus_size:
            raise ValueError("span_ids length must match BM25 index corpus_size.")

    def generate_candidates(
        self, claim: str, definition: RetrievalDefinition
    ) -> RetrievalCandidateSet:
        if not isinstance(definition, BM25RetrievalDefinition):
            raise RetrievalConfigurationError(
                f"BM25CandidateGenerator requires BM25RetrievalDefinition, got {type(definition).__name__}"
            )

        tokenized_query = self._tokenizer.tokenize(claim)
        scores = self._index.get_scores(tokenized_query)

        # Pair scores with original corpus index to preserve stable corpus ordering during tie-breaking.
        # We sort by: 1) Score descending, 2) Corpus index ascending.
        scored_items = [
            (float(score), idx) for idx, score in enumerate(scores)
        ]

        if definition.min_score is not None:
            scored_items = [
                item for item in scored_items if item[0] >= definition.min_score
            ]

        # Sort descending by score, ascending by original corpus index
        scored_items.sort(key=lambda x: (-x[0], x[1]))

        # Truncate
        top_items = scored_items[: definition.top_k]

        candidates = []
        for score, idx in top_items:
            candidates.append(
                RetrievalCandidate(
                    span_id=self._span_ids[idx],
                    score=score,
                    metadata={},
                )
            )

        return RetrievalCandidateSet(
            candidates=tuple(candidates),
            metadata=RetrievalMetadata(strategy_id="bm25", top_k=definition.top_k),
        )


class BM25Retriever(BaseRetriever):
    """
    Stateless concrete execution strategy for BM25 lexical retrieval orchestration.
    """

    def __init__(
        self,
        generator: BaseCandidateGenerator,
        document_store: DocumentStore,
    ) -> None:
        self._generator = generator
        self._document_store = document_store

    def validate_compatibility(self, definition: RetrievalDefinition) -> None:
        """Fails fast if the definition is not a BM25RetrievalDefinition."""
        if not isinstance(definition, BM25RetrievalDefinition):
            raise RetrievalConfigurationError(
                f"BM25Retriever requires BM25RetrievalDefinition, got {type(definition).__name__}"
            )

    def retrieve(self, claim: str, definition: RetrievalDefinition) -> EvidenceBundle:
        self.validate_compatibility(definition)

        if not isinstance(definition, BM25RetrievalDefinition):
            # for mypy narrowing
            raise RetrievalConfigurationError("Invalid definition type")

        try:
            candidate_set = self._generator.generate_candidates(claim, definition)

            passages = []
            for candidate in candidate_set.candidates:
                chunk = self._document_store.get_chunk(candidate.span_id)
                passages.append(
                    EvidencePassage(
                        document_id=chunk.document_id,
                        span_id=chunk.span_id,
                        text=chunk.text,
                        score=candidate.score,
                        metadata=candidate.metadata,
                    )
                )

            return EvidenceBundle(
                claim=claim,
                passages=tuple(passages),
                metadata=RetrievalMetadata(
                    strategy_id="bm25",
                    top_k=definition.top_k,
                ),
            )
        except Exception as e:
            if isinstance(e, (RetrievalExecutionError, RetrievalConfigurationError)):
                raise
            raise RetrievalExecutionError(f"BM25 retrieval execution failed: {e}") from e
