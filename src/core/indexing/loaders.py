import hashlib
import json
from collections.abc import Sequence

from src.core.indexing.pipeline import CorpusLoader
from src.core.retrieval.retrieval_models import CorpusEntry


class JSONLCorpusLoader(CorpusLoader):
    """Concrete loader for JSONL corpus files."""

    def load(self, corpus_path: str) -> tuple[str, Sequence[CorpusEntry]]:
        entries = []
        hasher = hashlib.sha256()
        with open(corpus_path, "rb") as f:
            for line in f:
                hasher.update(line)
                data = json.loads(line.decode("utf-8"))
                entries.append(
                    CorpusEntry(
                        document_id=data["document_id"],
                        span_id=data.get("span_id", data["document_id"]),
                        text=data["text"],
                    )
                )

        dataset_version = hasher.hexdigest()
        return dataset_version, entries
