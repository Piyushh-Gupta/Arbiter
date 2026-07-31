"""Stateless protocols for the Benchmarking subsystem."""

from typing import Any, Protocol, Sequence, runtime_checkable

from src.core.benchmark.benchmark_models import (
    BenchmarkDefinition,
    BenchmarkReport,
    MetricResult,
)


@runtime_checkable
class BaseBenchmark(Protocol):
    """Protocol for executing benchmark suites offline."""

    def run_benchmark(
        self,
        definition: BenchmarkDefinition,
    ) -> BenchmarkReport:
        """
        Executes a benchmark on a target pipeline.

        Args:
            definition: BenchmarkDefinition configuration.

        Returns:
            BenchmarkReport: Summarized metrics and traces.
        """
        ...


@runtime_checkable
class BaseBenchmarkDataset(Protocol):
    """Protocol for offline evaluation datasets."""

    def load_samples(self) -> tuple[dict[str, Any], ...]:
        """
        Loads samples from the dataset source.

        Returns:
            tuple[dict[str, Any], ...]: Extracted records containing query assertions.
        """
        ...

    def dataset_metadata(self) -> dict[str, Any]:
        """
        Returns dataset metadata attributes.

        Returns:
            dict[str, Any]: Metadata dictionary.
        """
        ...

    def dataset_version(self) -> str:
        """
        Returns semantic version tag of dataset.

        Returns:
            str: Version string.
        """
        ...


@runtime_checkable
class BaseMetricCalculator(Protocol):
    """Protocol for computing individual evaluation metrics."""

    def compute(
        self,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> MetricResult:
        """
        Computes the target metric from arrays of predictions and labels.

        Args:
            predictions: Predicted model output models/verdicts.
            ground_truths: Expected target ground truth values.

        Returns:
            MetricResult: Calculated metric container.
        """
        ...
