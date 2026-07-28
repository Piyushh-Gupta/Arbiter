import json
import os
import shutil
import tempfile
import typing

import pytest

from src.core.exceptions import RetrievalConfigurationError
from src.core.indexing.builders import (
    DenseIndexBuilder,
    MetadataIndexBuilder,
    SparseIndexBuilder,
)
from src.core.indexing.chunking import RecursiveDocumentChunker
from src.core.indexing.loaders import JSONLCorpusLoader
from src.core.indexing.models import ArtifactLocation, IndexManifest
from src.core.indexing.pipeline import IndexingPipeline
from src.core.indexing.validators import ManifestArtifactValidator
from src.core.indexing.writers import ManifestWriter
from tests.unit.test_retrieval_interfaces import DummyDocumentEncoder


@pytest.fixture
def temp_dir() -> typing.Generator[str, None, None]:
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def dummy_corpus(temp_dir: str) -> str:
    path = os.path.join(temp_dir, "corpus.jsonl")
    data = [
        {
            "document_id": "doc1",
            "text": "This is a test document. It has two sentences.",
        },
        {"document_id": "doc2", "text": "Another document here."},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return path


def test_corpus_loader(dummy_corpus: str) -> None:
    loader = JSONLCorpusLoader()
    version, entries = loader.load(dummy_corpus)
    assert len(entries) == 2
    assert version is not None
    assert entries[0].document_id == "doc1"


def test_recursive_chunker(dummy_corpus: str) -> None:
    loader = JSONLCorpusLoader()
    version, entries = loader.load(dummy_corpus)

    chunker = RecursiveDocumentChunker(max_length=15, overlap=5)
    chunks = chunker.chunk(entries, version)

    # doc1 is 46 chars, should be split into multiple chunks
    assert len(chunks) > 2
    assert all(c.dataset_version == version for c in chunks)


def test_pipeline_execution(dummy_corpus: str, temp_dir: str) -> None:
    loader = JSONLCorpusLoader()
    chunker = RecursiveDocumentChunker(max_length=50, overlap=10)
    encoder = DummyDocumentEncoder()
    builders = [SparseIndexBuilder(), DenseIndexBuilder(), MetadataIndexBuilder()]
    writer = ManifestWriter()

    version, _ = loader.load(dummy_corpus)
    validator = ManifestArtifactValidator(
        expected_dataset_version=version, expected_dimension=128
    )

    pipeline = IndexingPipeline(
        loader=loader,
        chunker=chunker,
        encoder=encoder,
        builders=builders,
        writer=writer,
        validator=validator,
    )

    manifest = pipeline.run(dummy_corpus, temp_dir)

    assert manifest.dataset_version == version
    assert manifest.embedding_dimension == 128
    assert "sparse_index" in manifest.artifacts
    assert "dense_index" in manifest.artifacts
    assert "metadata" in manifest.artifacts

    # Check physical files exist
    for artifact in manifest.artifacts.values():
        assert os.path.exists(artifact.path)


def test_manifest_validation_version_mismatch(temp_dir: str) -> None:
    manifest = IndexManifest(
        dataset_version="v1",
        encoder_model_id="test",
        embedding_dimension=128,
        artifacts={},
    )

    validator = ManifestArtifactValidator(
        expected_dataset_version="v2", expected_dimension=128
    )
    with pytest.raises(RetrievalConfigurationError, match="dataset version"):
        validator.validate(manifest)


def test_manifest_validation_dimension_mismatch(temp_dir: str) -> None:
    manifest = IndexManifest(
        dataset_version="v1",
        encoder_model_id="test",
        embedding_dimension=128,
        artifacts={},
    )

    validator = ManifestArtifactValidator(
        expected_dataset_version="v1", expected_dimension=256
    )
    with pytest.raises(RetrievalConfigurationError, match="embedding dimension"):
        validator.validate(manifest)


def test_manifest_validation_missing_artifact(temp_dir: str) -> None:
    manifest = IndexManifest(
        dataset_version="v1",
        encoder_model_id="test",
        embedding_dimension=128,
        artifacts={
            "fake": ArtifactLocation(
                path=os.path.join(temp_dir, "missing.bin"), checksum="123"
            )
        },
    )

    validator = ManifestArtifactValidator(
        expected_dataset_version="v1", expected_dimension=128
    )
    with pytest.raises(RetrievalConfigurationError, match="not found"):
        validator.validate(manifest)


def test_pipeline_reproducibility(dummy_corpus: str, temp_dir: str) -> None:
    dir1 = os.path.join(temp_dir, "run1")
    dir2 = os.path.join(temp_dir, "run2")
    os.makedirs(dir1)
    os.makedirs(dir2)

    loader = JSONLCorpusLoader()
    chunker = RecursiveDocumentChunker()
    encoder = DummyDocumentEncoder()
    builders = [SparseIndexBuilder(), DenseIndexBuilder(), MetadataIndexBuilder()]
    writer = ManifestWriter()

    version, _ = loader.load(dummy_corpus)
    validator = ManifestArtifactValidator(
        expected_dataset_version=version, expected_dimension=128
    )

    pipeline = IndexingPipeline(loader, chunker, encoder, builders, writer, validator)

    manifest1 = pipeline.run(dummy_corpus, dir1)
    manifest2 = pipeline.run(dummy_corpus, dir2)

    # The manifests should be identical in checksums
    assert (
        manifest1.artifacts["sparse_index"].checksum
        == manifest2.artifacts["sparse_index"].checksum
    )
    assert (
        manifest1.artifacts["dense_index"].checksum
        == manifest2.artifacts["dense_index"].checksum
    )
    assert (
        manifest1.artifacts["metadata"].checksum
        == manifest2.artifacts["metadata"].checksum
    )
