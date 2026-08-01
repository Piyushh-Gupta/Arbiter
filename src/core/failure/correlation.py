"""Concrete failure correlation strategies and base protocol interfaces (M3.4)."""

from typing import Protocol, runtime_checkable

from src.core.exceptions import FailureAnalysisConfigurationError
from src.core.failure.failure_models import (
    FailureCorrelation,
    FailureCorrelationContext,
    FailureCorrelationDefinition,
    FailureCorrelationResult,
)


@runtime_checkable
class BaseFailureCorrelationStrategy(Protocol):
    """Protocol for stateless execution of failure correlation logic."""

    def validate_compatibility(self, definition: FailureCorrelationDefinition) -> None:
        """Statically verifies compatibility of correlation configurations."""
        ...

    def correlate(self, context: FailureCorrelationContext) -> FailureCorrelationResult:
        """Evaluates rules over analyzer outputs to construct the dependency DAG."""
        ...


class DefaultFailureCorrelationStrategy(BaseFailureCorrelationStrategy):
    """Data-driven failure correlation engine evaluating FailureCorrelationRules statelessly."""

    def validate_compatibility(self, definition: FailureCorrelationDefinition) -> None:
        if not isinstance(definition, FailureCorrelationDefinition):
            raise FailureAnalysisConfigurationError(
                "DefaultFailureCorrelationStrategy requires FailureCorrelationDefinition."
            )

    def correlate(self, context: FailureCorrelationContext) -> FailureCorrelationResult:
        edges: list[FailureCorrelation] = []
        all_node_ids = [r.analyzer_id for r in context.analyzer_execution_results]

        # Evaluate rules to find correlation edges
        for rule in context.correlation_rules:
            if not rule.enabled:
                continue

            for source in context.analyzer_execution_results:
                if source.classification.category == rule.source_category:
                    for target in context.analyzer_execution_results:
                        if target.classification.category == rule.target_category:
                            edges.append(
                                FailureCorrelation(
                                    correlation_id=f"{rule.rule_id}_{source.analyzer_id}_to_{target.analyzer_id}",
                                    source_failure=source.analyzer_id,
                                    target_failure=target.analyzer_id,
                                    correlation_confidence=1.0,
                                )
                            )

        # Build adjacency list / dependency graph
        dependency_edges: dict[str, list[str]] = {nid: [] for nid in all_node_ids}
        in_degrees: dict[str, int] = {nid: 0 for nid in all_node_ids}

        for edge in edges:
            dependency_edges[edge.source_failure].append(edge.target_failure)
            in_degrees[edge.target_failure] += 1

        # Identify Root Failures (in-degree == 0)
        root_failures = [nid for nid, deg in in_degrees.items() if deg == 0]

        # Construct dependency_edges dict as tuple[str, ...] for immutable return
        dep_edges_tuples = {k: tuple(v) for k, v in dependency_edges.items()}

        summary = f"Failure correlation completed. Identified {len(edges)} edges and {len(root_failures)} root failures."

        return FailureCorrelationResult(
            correlation_graph=tuple(edges),
            root_failures=tuple(root_failures),
            dependency_edges=dep_edges_tuples,
            summary=summary,
        )
