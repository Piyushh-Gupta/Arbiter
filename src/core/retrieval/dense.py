from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.exceptions import RetrievalConfigurationError, RetrievalExecutionError
from src.core.indexing.models import IndexManifest
from src.core.retrieval.base import (
    BaseCandidateGenerator,
    BaseRetriever,
    BaseVectorStore,
    DocumentStore,
    QueryEncoder,
)
from src.core.retrieval.retrieval_models import (
    DenseRetrievalDefinition,
    EvidenceBundle,
    EvidencePassage,
    RetrievalCandidate,
    RetrievalCandidateSet,
    RetrievalDefinition,
    RetrievalMetadata,
)


class SentenceTransformerQueryEncoder(QueryEncoder):
    """Stateless query encoder using sentence-transformers."""

    def __init__(
        self, model_id: str, device: str = "cpu", normalize_embeddings: bool = True
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._normalize = normalize_embeddings
        self._model = SentenceTransformer(self._model_id, device=self._device)
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RetrievalConfigurationError(f"Could not determine dimension for {model_id}")
        self._dimension: int = dim
        self._pooling_strategy = "mean"
        self._normalization_strategy = "l2" if normalize_embeddings else "none"
        self._model_revision = None

    @property
    def pooling_strategy(self) -> str:
        return self._pooling_strategy
        
    @property
    def normalization_strategy(self) -> str:
        return self._normalization_strategy
        
    @property
    def model_revision(self) -> str | None:
        return self._model_revision

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    @property
    def device(self) -> str:
        return self._device

    def is_ready(self) -> bool:
        return self._model is not None

    def encode(self, text: str) -> np.ndarray:
        # SentenceTransformer output is usually a numpy array or torch tensor.
        # We enforce it to be numpy array.
        embeddings = self._model.encode(
            text, normalize_embeddings=self._normalize, show_progress_bar=False
        )
        return np.array(embeddings, dtype=np.float32)


class FAISSVectorStore(BaseVectorStore):
    """Stateless read-only vector store wrapping a FAISS index."""

    def __init__(
        self, index_path: str | Path, manifest: IndexManifest, span_ids: Sequence[str]
    ) -> None:
        self._manifest = manifest
        self._index = faiss.read_index(str(index_path))
        self._span_ids = tuple(span_ids)

        if not hasattr(self._manifest, "embedding_metadata"):
            raise RetrievalExecutionError("Manifest missing embedding_metadata.")

        expected_dim = self._manifest.embedding_metadata.embedding_dimension
        if self._index.d != expected_dim:
            raise RetrievalExecutionError(
                f"FAISS index dimension {self._index.d} does not match manifest {expected_dim}"
            )
            
        if self._index.ntotal != len(self._span_ids):
            raise RetrievalExecutionError(
                f"FAISS index size {self._index.ntotal} does not match span_ids length {len(self._span_ids)}"
            )

    def search(self, query: np.ndarray, top_k: int) -> tuple[RetrievalCandidate, ...]:
        query_batch = query.reshape(1, -1)
        distances, indices = self._index.search(query_batch, top_k)

        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            candidates.append(
                RetrievalCandidate(
                    span_id=self._span_ids[idx],
                    score=float(distances[0][i]),
                    metadata={"corpus_index": int(idx)},
                )
            )
        return tuple(candidates)


class DenseCandidateGenerator(BaseCandidateGenerator):
    """
    Stateless candidate generator for Dense semantic retrieval.
    """

    def __init__(
        self,
        query_encoder: QueryEncoder,
        vector_store: BaseVectorStore,
    ) -> None:
        self._query_encoder = query_encoder
        self._vector_store = vector_store

    def generate_candidates(
        self, claim: str, definition: RetrievalDefinition
    ) -> RetrievalCandidateSet:
        if not isinstance(definition, DenseRetrievalDefinition):
            raise RetrievalConfigurationError(
                f"DenseCandidateGenerator requires DenseRetrievalDefinition, got {type(definition).__name__}"
            )

        query_embedding = self._query_encoder.encode(claim)
        
        # We query for more if there is a threshold, but FAISS requires a k.
        # However, to be perfectly safe, we query the exact top_k requested, because min_score is just a filter.
        # Wait, if we filter, we might return fewer than top_k. But we can't search for "all above threshold" efficiently without a dynamic k.
        # Standard practice: search top_k, then filter.
        candidates = self._vector_store.search(query_embedding, definition.top_k)

        # Apply min_score threshold if present
        if definition.min_score is not None:
            candidates = tuple(
                c for c in candidates if c.score >= definition.min_score
            )

        # Sort deterministically:
        # 1. descending score
        # 2. ascending corpus index (to break ties stably). We stored corpus_index in metadata.
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (-c.score, c.metadata.get("corpus_index", 0)),
        )

        return RetrievalCandidateSet(
            candidates=tuple(sorted_candidates),
            metadata=RetrievalMetadata(strategy_id="dense", top_k=definition.top_k),
        )


class DenseRetriever(BaseRetriever):
    """
    Stateless orchestrator for dense retrieval.
    """

    def __init__(
        self,
        candidate_generator: BaseCandidateGenerator,
        document_store: DocumentStore,
    ) -> None:
        self._candidate_generator = candidate_generator
        self._document_store = document_store

    def validate_compatibility(self, definition: RetrievalDefinition) -> None:
        if not isinstance(definition, DenseRetrievalDefinition):
            raise RetrievalConfigurationError(
                f"DenseRetriever requires DenseRetrievalDefinition, got {type(definition).__name__}"
            )

    def retrieve(self, claim: str, definition: RetrievalDefinition) -> EvidenceBundle:
        self.validate_compatibility(definition)

        candidate_set = self._candidate_generator.generate_candidates(claim, definition)

        passages = []
        for candidate in candidate_set.candidates:
            chunk = self._document_store.get_chunk(candidate.span_id)
            passages.append(
                EvidencePassage(
                    document_id=chunk.document_id,
                    span_id=chunk.span_id,
                    text=chunk.text,
                    score=candidate.score,
                    metadata=chunk.metadata,
                )
            )

        return EvidenceBundle(
            claim=claim,
            passages=tuple(passages),
            metadata=candidate_set.metadata,
        )
