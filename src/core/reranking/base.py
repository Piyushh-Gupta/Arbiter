"""Stateless base reranking protocol."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class CrossEncoderScorer(Protocol):
    """Stateless protocol for cross-encoder scoring."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """
        Scores a sequence of passages against a query.

        Receives:
        - query: The claim text.
        - passages: The raw passage texts to score.

        Returns:
        - list[float]: Relevance scores, matching the order of the input passages.
          Must return exactly len(passages) scores.

        The scorer is fully responsible for model lifecycle, device placement,
        numerical normalization, and internal batching.
        """
        ...
