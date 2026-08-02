"""Production Failure Analysis Optimization & Hardening subsystem (M3.8)."""

from src.core.failure.optimization.controller import FailureOptimizationController
from src.core.failure.optimization.health import FailureHealthMonitor
from src.core.failure.optimization.implementations import (
    BoundedSemaphoreConcurrencyLimiter,
    FailureTelemetryCollector,
)
from src.core.failure.optimization.optimization_models import (
    FailureExecutionMetrics,
    FailureOperationalProfile,
    FailureOptimizationDefinition,
    FailureOptimizationProfile,
    FailureOptimizationProfileRegistry,
    FailureTelemetryRecord,
    FailureTelemetrySnapshot,
)

__all__ = [
    "BoundedSemaphoreConcurrencyLimiter",
    "FailureExecutionMetrics",
    "FailureHealthMonitor",
    "FailureOperationalProfile",
    "FailureOptimizationController",
    "FailureOptimizationDefinition",
    "FailureOptimizationProfile",
    "FailureOptimizationProfileRegistry",
    "FailureTelemetryCollector",
    "FailureTelemetryRecord",
    "FailureTelemetrySnapshot",
]
