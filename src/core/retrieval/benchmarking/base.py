"""Stateless protocols for the Retrieval Benchmarking & Evaluation subsystem."""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from src.core.retrieval.benchmarking.benchmark_models import (
    BenchmarkDataset,
    BenchmarkDefinition,
    BenchmarkEnvironmentMetadata,
    BenchmarkReport,
)
from src.core.retrieval.retrieval_models import EvidencePassage


@runtime_checkable
class MetricCalculator(Protocol):
    """Stateless protocol for pure metric calculation functions."""

    def calculate(
        self,
        retrieved_passages: Sequence[EvidencePassage],
        expected_span_ids: Sequence[str],
        expected_doc_ids: Sequence[str] = (),
        top_k: int = 5,
    ) -> float:
        """
        Calculates a single scalar metric score.
        """
        ...


@runtime_checkable
class BaseRetrievalBenchmark(Protocol):
    """Stateless protocol for retrieval benchmark evaluation engines."""

    def validate_compatibility(self, definition: BenchmarkDefinition) -> None:
        """Fails fast if definition is incompatible with the evaluator."""
        ...

    def evaluate_dataset(
        self,
        dataset: BenchmarkDataset,
        retriever: Any,
        definition: BenchmarkDefinition,
        reranker: Any | None = None,
        retrieval_definition: Any | None = None,
        reranking_definition: Any | None = None,
        environment_metadata: BenchmarkEnvironmentMetadata | None = None,
    ) -> BenchmarkReport:
        """
        Executes offline benchmark evaluation over a dataset and returns an immutable BenchmarkReport.
        """
        ...
