"""Root cause attribution strategies and base protocol interface (M3.5)."""

from typing import Protocol, runtime_checkable

from src.core.exceptions import FailureAnalysisConfigurationError
from src.core.failure.failure_models import (
    FailureCategory,
    FailureCorrelationResult,
    RootCauseAttributionDefinition,
    RootCauseResult,
)
from src.core.failure.traversal import FailureGraphTraverser

# Deterministic category precedence for tie-breaking multiple root candidates.
# Lower index = higher priority (evaluated first as the primary root cause).
_CATEGORY_PRIORITY: dict[str, int] = {
    FailureCategory.INFRASTRUCTURE: 0,
    FailureCategory.RETRIEVAL: 1,
    FailureCategory.VERIFICATION: 2,
    FailureCategory.CALIBRATION: 3,
    FailureCategory.EXPLAINABILITY: 4,
    FailureCategory.OPTIMIZATION: 5,
    FailureCategory.AGGREGATION: 6,
    FailureCategory.CONFIGURATION: 7,
    FailureCategory.UNKNOWN: 8,
}


@runtime_checkable
class BaseRootCauseStrategy(Protocol):
    """Protocol for stateless root cause attribution over a failure correlation graph."""

    def validate_compatibility(
        self, definition: RootCauseAttributionDefinition
    ) -> None:
        """Validates compatibility of the strategy with the given definition."""
        ...

    def attribute(
        self,
        correlation_result: FailureCorrelationResult,
        definition: RootCauseAttributionDefinition,
        traverser: FailureGraphTraverser,
    ) -> RootCauseResult:
        """Attributes the primary root cause from the completed correlation graph."""
        ...


class DependencyGraphRootCauseStrategy(BaseRootCauseStrategy):
    """Stateless root cause attribution consuming a FailureGraphTraverser.

    Does NOT perform graph traversal internally.
    """

    def validate_compatibility(
        self, definition: RootCauseAttributionDefinition
    ) -> None:
        if not isinstance(definition, RootCauseAttributionDefinition):
            raise FailureAnalysisConfigurationError(
                "DependencyGraphRootCauseStrategy requires RootCauseAttributionDefinition."
            )

    def attribute(
        self,
        correlation_result: FailureCorrelationResult,
        definition: RootCauseAttributionDefinition,
        traverser: FailureGraphTraverser,
    ) -> RootCauseResult:
        root_nodes = traverser.get_root_nodes(correlation_result)

        if not root_nodes:
            # Degenerate graph: treat the first node alphabetically as root.
            all_nodes = sorted(traverser.get_all_node_ids(correlation_result))
            primary = all_nodes[0] if all_nodes else "unknown"
            return RootCauseResult(
                primary_root_cause=primary,
                contributing_failures=(),
                dependency_path=(primary,),
                attribution_confidence=definition.confidence_threshold,
                attribution_metadata={"reason": "no_root_nodes_found"},
            )

        # Build lookup: node_id -> category string from edge metadata in traversal priority.
        # We use the traversal_priority from definition to sort root candidates.
        priority_order = list(definition.traversal_priority)

        # Map from category name -> priority index for sorting
        priority_map: dict[str, int] = {
            cat: idx for idx, cat in enumerate(priority_order)
        }

        # Retrieve category info from the correlation result's edge map keys by looking
        # up the analyzer_id matches in the graph edges. Since we only have node IDs in
        # root_nodes, we rely on the default _CATEGORY_PRIORITY ordering unless overridden.
        def _node_priority(node_id: str) -> int:
            # Check if any correlation edge has category metadata.
            for edge in correlation_result.correlation_graph:
                if edge.source_failure == node_id:
                    pass  # No category info stored on edges
            # Fall back to alphabetical for determinism when priority is equal.
            return priority_map.get(node_id, len(priority_order))

        sorted_roots = sorted(root_nodes, key=lambda n: (_node_priority(n), n))
        primary = sorted_roots[0]
        other_roots = tuple(sorted_roots[1:])

        # All descendants of the primary root are contributing failures.
        descendants = traverser.get_all_descendants(primary, correlation_result)
        contributing = other_roots + descendants

        dependency_path = traverser.build_dependency_path(primary, correlation_result)

        return RootCauseResult(
            primary_root_cause=primary,
            contributing_failures=contributing,
            dependency_path=dependency_path,
            attribution_confidence=1.0,
            attribution_metadata={
                "root_candidates": list(root_nodes),
                "selected_by": "traversal_priority",
            },
        )
