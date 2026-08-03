"""Base protocols for Decision Benchmarking & Evaluation Framework (M4.5)."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BaseDecisionMetricCalculator(Protocol):
    """Stateless protocol for pluggable decision metric calculations."""

    @property
    def metric_name(self) -> str:
        """Unique identifier for the benchmark metric."""
        ...

    def calculate(self, raw_output: Any) -> float:
        """Computes metric value based on raw benchmark execution outputs."""
        ...


@runtime_checkable
class BaseDecisionBenchmarkExecutor(Protocol):
    """Stateless protocol responsible only for executing benchmark dataset and producing raw outputs."""

    def execute(self, dataset: Any, strategy: Any) -> Any:
        """Executes decision strategy over all dataset items and returns raw outputs."""
        ...


@runtime_checkable
class BaseDecisionBenchmarkRunner(Protocol):
    """Stateless protocol responsible for orchestrating benchmark runner execution."""

    def run(self, suite: Any, profile: Any) -> Any:
        """Orchestrates executor, metric engine, and outputs benchmark result."""
        ...
