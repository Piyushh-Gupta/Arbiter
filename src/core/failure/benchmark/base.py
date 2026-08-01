"""Base protocols for the Failure Analysis Benchmarking subsystem (M3.6)."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BaseFailureBenchmarkDataset(Protocol):
    """Protocol defining the interface for an offline failure benchmark dataset."""

    @property
    def name(self) -> str:
        """Human-readable dataset name."""
        ...

    @property
    def version(self) -> str:
        """Dataset version string."""
        ...

    @property
    def description(self) -> str:
        """Short description of the dataset contents."""
        ...

    def items(self) -> tuple[Any, ...]:
        """Return all benchmark items in deterministic order."""
        ...


@runtime_checkable
class BaseMetricCalculator(Protocol):
    """Protocol for a stateless benchmark metric calculator."""

    @property
    def metric_name(self) -> str:
        """Unique metric identifier."""
        ...

    def calculate(self, results: Any) -> float:
        """Compute the metric from the given benchmark results."""
        ...


@runtime_checkable
class BaseFailureBenchmarkRunner(Protocol):
    """Protocol for a stateless failure benchmark runner."""

    def run(self, suite: Any, definition: Any) -> Any:
        """Execute a benchmark suite and return an immutable BenchmarkReport."""
        ...
