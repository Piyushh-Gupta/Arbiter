import hashlib
import os
from collections.abc import Sequence

import numpy as np

from src.core.indexing.models import Chunk
from src.core.indexing.pipeline import IndexBuilder
from src.core.retrieval.base import Tokenizer


def _hash_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class SparseIndexBuilder(IndexBuilder):
    def __init__(self, tokenizer: "Tokenizer") -> None:
        self._tokenizer = tokenizer

    @property
    def builder_id(self) -> str:
        return "sparse_index"

    def build(
        self,
        chunks: Sequence[Chunk],
        embeddings: np.ndarray | None,
        output_dir: str,
    ) -> tuple[str, str]:
        import pickle

        from rank_bm25 import BM25Okapi  # type: ignore

        tokenized_corpus = [self._tokenizer.tokenize(chunk.text) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        output_path = os.path.join(output_dir, "bm25_index.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(bm25, f)

        return output_path, _hash_file(output_path)


class DenseIndexBuilder(IndexBuilder):
    @property
    def builder_id(self) -> str:
        return "dense_index"

    def build(
        self,
        chunks: Sequence[Chunk],
        embeddings: np.ndarray | None,
        output_dir: str,
    ) -> tuple[str, str]:
        import faiss

        if embeddings is None:
            raise ValueError("DenseIndexBuilder requires embeddings")

        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)

        output_path = os.path.join(output_dir, "faiss_index.bin")
        faiss.write_index(index, output_path)

        return output_path, _hash_file(output_path)


class MetadataIndexBuilder(IndexBuilder):
    @property
    def builder_id(self) -> str:
        return "metadata"

    def build(
        self,
        chunks: Sequence[Chunk],
        embeddings: np.ndarray | None,
        output_dir: str,
    ) -> tuple[str, str]:
        output_path = os.path.join(output_dir, "metadata.jsonl")

        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")

        return output_path, _hash_file(output_path)
