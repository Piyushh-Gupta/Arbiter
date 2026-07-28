from collections.abc import Sequence

from src.core.indexing.models import Chunk
from src.core.indexing.pipeline import DocumentChunker
from src.core.retrieval.retrieval_models import CorpusEntry


class RecursiveDocumentChunker(DocumentChunker):
    """Concrete implementation of recursive chunking."""

    def __init__(self, max_length: int = 1000, overlap: int = 100) -> None:
        self.max_length = max_length
        self.overlap = overlap

    def chunk(
        self, corpus: Sequence[CorpusEntry], dataset_version: str
    ) -> Sequence[Chunk]:
        chunks = []
        for entry in corpus:
            text = entry.text
            if len(text) <= self.max_length:
                chunks.append(
                    Chunk(
                        span_id=f"{entry.document_id}-0",
                        document_id=entry.document_id,
                        text=text,
                        start_char=0,
                        end_char=len(text),
                        dataset_version=dataset_version,
                        metadata={},
                    )
                )
                continue

            start = 0
            chunk_idx = 0
            while start < len(text):
                end = min(start + self.max_length, len(text))
                # If we're not at the end of the text, try to find a natural break
                if end < len(text):
                    # Try to break on paragraph
                    break_idx = text.rfind("\n\n", start, end)
                    if break_idx == -1:
                        # Try to break on sentence
                        break_idx = text.rfind(". ", start, end)
                    if break_idx == -1:
                        # Try to break on space
                        break_idx = text.rfind(" ", start, end)

                    if break_idx != -1 and break_idx > start:
                        end = break_idx + 1  # Include the break character

                chunk_text = text[start:end]
                chunks.append(
                    Chunk(
                        span_id=f"{entry.document_id}-{chunk_idx}",
                        document_id=entry.document_id,
                        text=chunk_text,
                        start_char=start,
                        end_char=end,
                        dataset_version=dataset_version,
                        metadata={},
                    )
                )
                chunk_idx += 1

                # Advance start, but move back by overlap if we haven't reached the end
                if end < len(text):
                    next_start = end - self.overlap
                    if next_start <= start:
                        next_start = end  # prevent infinite loop by guaranteeing strictly increasing start
                    start = next_start
                else:
                    break

        return chunks
