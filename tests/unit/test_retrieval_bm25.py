import os
import shutil
import tempfile
import typing

import pytest
from rank_bm25 import BM25Okapi  # type: ignore

from src.core.exceptions import RetrievalConfigurationError, RetrievalExecutionError
from src.core.indexing.models import Chunk
from src.core.retrieval.bm25 import (
    BM25CandidateGenerator,
    BM25Retriever,
    MetadataDocumentStore,
    WhitespaceTokenizer,
)
from src.core.retrieval.retrieval_models import (
    BM25RetrievalDefinition,
    DenseRetrievalDefinition,
)


@pytest.fixture
def temp_dir() -> typing.Generator[str, None, None]:
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def dummy_metadata_path(temp_dir: str) -> str:
    path = os.path.join(temp_dir, "metadata.jsonl")
    chunks = [
        Chunk(
            span_id="doc1-0",
            document_id="doc1",
            text="hello world",
            start_char=0,
            end_char=11,
            dataset_version="1.0",
            metadata={},
        ),
        Chunk(
            span_id="doc2-0",
            document_id="doc2",
            text="hello arbiter",
            start_char=0,
            end_char=13,
            dataset_version="1.0",
            metadata={},
        ),
        Chunk(
            span_id="doc3-0",
            document_id="doc3",
            text="world of arbiter",
            start_char=0,
            end_char=16,
            dataset_version="1.0",
            metadata={},
        ),
        Chunk(
            span_id="doc4-0",
            document_id="doc4",
            text="unrelated document",
            start_char=0,
            end_char=18,
            dataset_version="1.0",
            metadata={},
        ),
        Chunk(
            span_id="doc5-0",
            document_id="doc5",
            text="another random text",
            start_char=0,
            end_char=19,
            dataset_version="1.0",
            metadata={},
        ),
    ]
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    return path


@pytest.fixture
def dummy_bm25_index() -> BM25Okapi:
    corpus = [
        ["hello", "world"],
        ["hello", "arbiter"],
        ["world", "of", "arbiter"],
        ["unrelated", "document"],
        ["another", "random", "text"],
    ]
    return BM25Okapi(corpus)


def test_whitespace_tokenizer() -> None:
    tokenizer = WhitespaceTokenizer()
    assert tokenizer.tokenize("Hello World! ") == ["hello", "world!"]


def test_metadata_document_store(dummy_metadata_path: str) -> None:
    store = MetadataDocumentStore(dummy_metadata_path)
    chunk = store.get_chunk("doc2-0")
    assert chunk.text == "hello arbiter"

    with pytest.raises(RetrievalExecutionError):
        store.get_chunk("missing")


def test_bm25_generator(dummy_bm25_index: BM25Okapi) -> None:
    tokenizer = WhitespaceTokenizer()
    span_ids = ["doc1-0", "doc2-0", "doc3-0", "doc4-0", "doc5-0"]
    generator = BM25CandidateGenerator(dummy_bm25_index, span_ids, tokenizer)

    definition = BM25RetrievalDefinition(top_k=2)
    candidates = generator.generate_candidates("hello", definition)

    assert len(candidates.candidates) == 2
    assert candidates.candidates[0].span_id == "doc1-0"
    assert candidates.candidates[1].span_id == "doc2-0"

    # Test score filter
    definition_filtered = BM25RetrievalDefinition(top_k=5, min_score=10.0)
    candidates_filtered = generator.generate_candidates("hello", definition_filtered)
    assert len(candidates_filtered.candidates) == 0

    # Test invalid definition
    with pytest.raises(RetrievalConfigurationError):
        generator.generate_candidates("hello", DenseRetrievalDefinition(top_k=5))


def test_bm25_retriever(dummy_bm25_index: BM25Okapi, dummy_metadata_path: str) -> None:
    tokenizer = WhitespaceTokenizer()
    span_ids = ["doc1-0", "doc2-0", "doc3-0", "doc4-0", "doc5-0"]
    generator = BM25CandidateGenerator(dummy_bm25_index, span_ids, tokenizer)
    store = MetadataDocumentStore(dummy_metadata_path)
    retriever = BM25Retriever(generator, store)

    definition = BM25RetrievalDefinition(top_k=2)

    # validate compatibility
    retriever.validate_compatibility(definition)
    with pytest.raises(RetrievalConfigurationError):
        retriever.validate_compatibility(DenseRetrievalDefinition(top_k=5))

    bundle = retriever.retrieve("world", definition)
    assert bundle.metadata.strategy_id == "bm25"
    assert bundle.metadata.top_k == 2
    assert len(bundle.passages) == 2
    assert bundle.passages[0].span_id == "doc1-0"
    assert bundle.passages[0].text == "hello world"


def test_bm25_determinism(dummy_bm25_index: BM25Okapi) -> None:
    tokenizer = WhitespaceTokenizer()
    span_ids = ["doc1-0", "doc2-0", "doc3-0", "doc4-0", "doc5-0"]
    generator = BM25CandidateGenerator(dummy_bm25_index, span_ids, tokenizer)
    definition = BM25RetrievalDefinition(top_k=3)

    first_candidates = generator.generate_candidates("hello arbiter", definition)

    for _ in range(10):
        c = generator.generate_candidates("hello arbiter", definition)
        assert c == first_candidates
