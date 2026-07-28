"""Stateless protocols and execution orchestration for the Offline Indexing Framework."""

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from src.core.indexing.models import Chunk, IndexManifest
from src.core.retrieval.base import DocumentEncoder
from src.core.retrieval.retrieval_models import CorpusEntry


@runtime_checkable
class CorpusLoader(Protocol):
    """Protocol for loading raw datasets into immutable corpus entries."""

    def load(self, corpus_path: str) -> tuple[str, Sequence[CorpusEntry]]:
        """Loads a corpus and returns (dataset_version_checksum, entries)."""
        ...


@runtime_checkable
class DocumentChunker(Protocol):
    """Protocol for chunking corpus entries into indexable chunks."""

    def chunk(
        self, corpus: Sequence[CorpusEntry], dataset_version: str
    ) -> Sequence[Chunk]:
        """Splits corpus entries into a deterministic sequence of immutable Chunks."""
        ...


@runtime_checkable
class IndexBuilder(Protocol):
    """Protocol for specialized index generation (Sparse, Dense, Metadata)."""

    @property
    def builder_id(self) -> str:
        """Identifier for the builder (e.g. 'sparse', 'dense', 'metadata')."""
        ...

    def build(
        self,
        chunks: Sequence[Chunk],
        embeddings: np.ndarray | None,
        output_dir: str,
    ) -> tuple[str, str]:
        """
        Builds the index and persists it to output_dir.
        Returns a tuple of (artifact_path, checksum).
        """
        ...


@runtime_checkable
class MetadataWriter(Protocol):
    """Protocol for finalizing and writing the IndexManifest."""

    def write_manifest(
        self, manifest: IndexManifest, output_dir: str
    ) -> tuple[str, str]:
        """
        Persists the manifest to disk.
        Returns (manifest_path, checksum).
        """
        ...


@runtime_checkable
class ArtifactValidator(Protocol):
    """Protocol for validating generated indexing artifacts."""

    def validate(self, manifest: IndexManifest) -> None:
        """
        Verifies checksums, versions, and embedding dimensions fail-fast.
        """
        ...


class IndexingPipeline:
    def __init__(
        self,
        loader: CorpusLoader,
        chunker: DocumentChunker,
        encoder: DocumentEncoder,
        builders: Sequence[IndexBuilder],
        writer: MetadataWriter,
        validator: ArtifactValidator,
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._encoder = encoder
        self._builders = tuple(builders)
        self._writer = writer
        self._validator = validator

    def run(self, corpus_path: str, output_dir: str) -> IndexManifest:
        dataset_version, corpus = self._loader.load(corpus_path)
        chunks = self._chunker.chunk(corpus, dataset_version)
        # Extract text for encoding
        texts = [c.text for c in chunks]
        embeddings = self._encoder.encode_batch(texts) if texts else None

        from src.core.indexing.models import ArtifactLocation, EmbeddingModelMetadata, IndexManifest

        artifacts = {}
        for builder in self._builders:
            path, checksum = builder.build(chunks, embeddings, output_dir)
            artifacts[builder.builder_id] = ArtifactLocation(
                path=path, checksum=checksum
            )

        manifest = IndexManifest(
            dataset_version=dataset_version,
            embedding_metadata=EmbeddingModelMetadata(
                model_id=self._encoder.model_id,
                embedding_dimension=self._encoder.embedding_dimension,
                pooling_strategy=self._encoder.pooling_strategy,
                normalization_strategy=self._encoder.normalization_strategy,
                model_revision=self._encoder.model_revision,
            ),
            artifacts=artifacts,
        )
        self._writer.write_manifest(manifest, output_dir)
        self._validator.validate(manifest)
        return manifest
