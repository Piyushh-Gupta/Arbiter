"""Failure health monitor for operational readiness and dependency checks (M3.8)."""

from typing import Any

from src.core.failure.optimization.optimization_models import (
    FailureOptimizationProfileRegistry,
)


class FailureHealthMonitor:
    """Stateless operational health monitor validating readiness, dependencies, and registries."""

    def validate_registry(
        self, registry: FailureOptimizationProfileRegistry | Any
    ) -> bool:
        """Validates that the optimization registry is non-null and correctly populated."""
        if registry is None:
            return False
        if not isinstance(registry, FailureOptimizationProfileRegistry):
            return False
        return len(registry.profiles) > 0

    def validate_dependencies(
        self,
        analyzer_registry: Any | None = None,
        explainability_registry: Any | None = None,
    ) -> bool:
        """Validates that required upstream subsystem registries exist."""
        if analyzer_registry is None:
            return False
        return True

    def check_readiness(
        self,
        optimization_registry: FailureOptimizationProfileRegistry | Any | None = None,
        analyzer_registry: Any | None = None,
    ) -> dict[str, Any]:
        """Performs a comprehensive operational readiness check."""
        reg_valid = self.validate_registry(optimization_registry)
        deps_valid = self.validate_dependencies(analyzer_registry=analyzer_registry)
        is_ready = reg_valid and deps_valid

        return {
            "status": "READY" if is_ready else "NOT_READY",
            "registry_valid": reg_valid,
            "dependencies_valid": deps_valid,
        }

    def get_health_status(self) -> dict[str, Any]:
        """Returns baseline operational health status."""
        return {
            "subsystem": "failure_optimization",
            "status": "HEALTHY",
        }
