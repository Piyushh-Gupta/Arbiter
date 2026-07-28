import numpy as np

from src.core.retrieval.base import (
    BaseCandidateGenerator,
    BaseEncoder,
    BaseReranker,
    BaseVectorStore,
    DocumentEncoder,
    IndexBuilder,
    QueryEncoder,
)
from src.core.retrieval.retrieval_models import (
    RetrievalCandidate,
    RetrievalCandidateSet,
    RetrievalDefinition,
    RetrievalMetadata,
)


class DummyQueryEncoder:
    @property
    def model_id(self) -> str:
        return "dummy-query"

    @property
    def embedding_dimension(self) -> int:
        return 128

    @property
    def device(self) -> str:
        return "cpu"

    def is_ready(self) -> bool:
        return True

    def encode(self, text: str) -> np.ndarray:
        return np.zeros(128, dtype=np.float32)


class DummyDocumentEncoder:
    @property
    def model_id(self) -> str:
        return "dummy-doc"

    @property
    def embedding_dimension(self) -> int:
        return 128

    @property
    def device(self) -> str:
        return "cpu"

    def is_ready(self) -> bool:
        return True

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 128), dtype=np.float32)


class DummyVectorStore:
    def search(self, query: np.ndarray, top_k: int) -> tuple[RetrievalCandidate, ...]:
        return (RetrievalCandidate(span_id="span1", score=1.0, metadata={}),)


class DummyCandidateGenerator:
    def generate_candidates(
        self, claim: str, definition: RetrievalDefinition
    ) -> RetrievalCandidateSet:
        return RetrievalCandidateSet(
            candidates=(RetrievalCandidate(span_id="span1", score=1.0, metadata={}),),
            metadata=RetrievalMetadata(strategy_id="dummy", top_k=1),
        )


class DummyReranker:
    def rerank(
        self,
        claim: str,
        candidates: RetrievalCandidateSet,
        definition: RetrievalDefinition,
    ) -> RetrievalCandidateSet:
        return candidates


class DummyIndexBuilder:
    def build_index(self, corpus_path: str, index_output_path: str) -> None:
        pass


def test_query_encoder_interface_compliance() -> None:
    encoder = DummyQueryEncoder()
    assert isinstance(encoder, BaseEncoder)
    assert isinstance(encoder, QueryEncoder)
    assert encoder.model_id == "dummy-query"
    assert encoder.embedding_dimension == 128
    assert encoder.device == "cpu"
    assert encoder.is_ready() is True


def test_document_encoder_interface_compliance() -> None:
    encoder = DummyDocumentEncoder()
    assert isinstance(encoder, BaseEncoder)
    assert isinstance(encoder, DocumentEncoder)
    assert encoder.model_id == "dummy-doc"
    assert encoder.embedding_dimension == 128
    assert encoder.device == "cpu"
    assert encoder.is_ready() is True


def test_vector_store_interface_compliance() -> None:
    store = DummyVectorStore()
    assert isinstance(store, BaseVectorStore)
    results = store.search(np.zeros(128), 1)
    assert len(results) == 1
    assert isinstance(results[0], RetrievalCandidate)


def test_candidate_generator_interface_compliance() -> None:
    generator = DummyCandidateGenerator()
    assert isinstance(generator, BaseCandidateGenerator)
    result = generator.generate_candidates("claim", RetrievalDefinition())
    assert isinstance(result, RetrievalCandidateSet)


def test_reranker_interface_compliance() -> None:
    reranker = DummyReranker()
    assert isinstance(reranker, BaseReranker)
    candidates = RetrievalCandidateSet(
        candidates=(), metadata=RetrievalMetadata(strategy_id="test", top_k=0)
    )
    result = reranker.rerank("claim", candidates, RetrievalDefinition())
    assert isinstance(result, RetrievalCandidateSet)


def test_index_builder_interface_compliance() -> None:
    builder = DummyIndexBuilder()
    assert isinstance(builder, IndexBuilder)
