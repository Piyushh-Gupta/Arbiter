"""Base protocols for Pipeline Benchmarking & Evaluation Framework (M5.4)."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BaseClock(Protocol):
    """Abstract interface for measuring elapsed time."""

    def now_ms(self) -> float:
        """Returns the current monotonic time in milliseconds."""
        ...


@runtime_checkable
class BasePipelineMetricCalculator(Protocol):
    """Stateless protocol for pipeline benchmark metric computation."""

    @property
    def metric_name(self) -> str:
        """Unique metric identifier matching a field on PipelineBenchmarkMetrics."""
        ...

    def calculate(self, raw_output: Any) -> float:
        """Computes a single scalar metric from raw benchmark output."""
        ...


@runtime_checkable
class BasePipelineBenchmarkExecutor(Protocol):
    """Stateless protocol for executing a benchmark dataset through the pipeline."""

    def execute(
        self,
        dataset: Any,
        orchestrator: Any,
    ) -> Any:
        """Drives orchestrator.execute() for every dataset item and returns raw outputs."""
        ...


@runtime_checkable
class BasePipelineBenchmarkRunner(Protocol):
    """Stateless protocol for orchestrating executor, metric engine, and report building."""

    def run(
        self,
        suite: Any,
        profile: Any,
    ) -> Any:
        """Orchestrates a complete benchmark run and produces an immutable report."""
        ...
