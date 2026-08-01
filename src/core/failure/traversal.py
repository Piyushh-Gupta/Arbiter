"""Stateless graph traversal utility for failure dependency DAGs (M3.5)."""

from src.core.failure.failure_models import FailureCorrelationResult


class FailureGraphTraverser:
    """Stateless utility for traversing failure correlation dependency graphs.

    Contains no attribution logic. Provides graph inspection primitives only.
    """

    def get_root_nodes(
        self, correlation_result: FailureCorrelationResult
    ) -> tuple[str, ...]:
        """Return all node IDs that have no inbound edges (in-degree == 0)."""
        return correlation_result.root_failures

    def get_downstream_nodes(
        self, node_id: str, correlation_result: FailureCorrelationResult
    ) -> tuple[str, ...]:
        """Return all nodes directly downstream from the given node."""
        return correlation_result.dependency_edges.get(node_id, ())

    def get_all_descendants(
        self, node_id: str, correlation_result: FailureCorrelationResult
    ) -> tuple[str, ...]:
        """Return all transitive descendants of the given node via BFS."""
        visited: list[str] = []
        queue = list(correlation_result.dependency_edges.get(node_id, ()))
        while queue:
            current = queue.pop(0)
            if current not in visited:
                visited.append(current)
                queue.extend(correlation_result.dependency_edges.get(current, ()))
        return tuple(visited)

    def build_dependency_path(
        self, root_id: str, correlation_result: FailureCorrelationResult
    ) -> tuple[str, ...]:
        """Construct the deterministic linear path from a root node through its descendants."""
        path: list[str] = [root_id]
        current = root_id
        visited: set[str] = {root_id}
        while True:
            children = correlation_result.dependency_edges.get(current, ())
            # Pick the first unvisited child for a deterministic linear path
            next_node = next((c for c in children if c not in visited), None)
            if next_node is None:
                break
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        return tuple(path)

    def get_all_node_ids(
        self, correlation_result: FailureCorrelationResult
    ) -> tuple[str, ...]:
        """Return all node IDs present in the correlation graph."""
        return tuple(correlation_result.dependency_edges.keys())
