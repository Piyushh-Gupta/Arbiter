"""Pytest configuration and fixtures."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.core.paths import ProjectPaths


@pytest.fixture(autouse=True)
def mock_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        import src.core.retrieval.dense

        mock_st = MagicMock()
        mock_st.return_value.get_sentence_embedding_dimension.return_value = 128
        mock_st.return_value.encode.return_value = np.zeros((1, 128), dtype=np.float32)
        monkeypatch.setattr(src.core.retrieval.dense, "SentenceTransformer", mock_st)
    except ImportError:
        pass


@pytest.fixture(autouse=True, scope="session")
def setup_dummy_index_for_tests(tmp_path_factory: pytest.TempPathFactory) -> None:
    from src.cli.index import build

    tmp_path = tmp_path_factory.mktemp("dummy_index")
    corpus_path = tmp_path / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(
            json.dumps({"document_id": "doc1", "text": "hello world", "metadata": {}})
            + "\n"
        )

    index_dir = tmp_path / "index"
    index_dir.mkdir()

    build(str(corpus_path), str(index_dir))

    # Overwrite the ProjectPaths.DATA_INDEX class attribute
    # Since this is a session fixture, it will apply to all tests.
    ProjectPaths.DATA_INDEX = index_dir
