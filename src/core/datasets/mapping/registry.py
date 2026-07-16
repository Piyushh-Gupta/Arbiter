"""Registry for resolving schema mappings."""

from src.core.datasets.mapping.base import TaskMapping
from src.core.datasets.mapping_models import TaskSchemaType
from src.core.exceptions import UnsupportedTaskSchemaError


class MappingRegistry:
    """
    Immutable runtime registry for schema mapping resolution.

    Responsible exclusively for registering and resolving unified TaskMapping objects.
    """

    def __init__(self) -> None:
        self._mappings: dict[TaskSchemaType, TaskMapping] = {}
        self._frozen = False

    def register(self, task_type: TaskSchemaType, task_mapping: TaskMapping) -> None:
        """Registers a TaskMapping for a given TaskSchemaType."""
        if self._frozen:
            raise RuntimeError("Cannot register mappings after initialization.")
        self._mappings[task_type] = task_mapping

    def freeze(self) -> None:
        """Locks the registry to prevent further registration."""
        self._frozen = True

    def resolve(self, task_type: TaskSchemaType) -> TaskMapping:
        """
        Resolves the appropriate TaskMapping for a requested task.

        Args:
            task_type: The requested task schema type.

        Returns:
            TaskMapping: The unified mapping containing the definition and mapper.

        Raises:
            UnsupportedTaskSchemaError: If the task type is not registered.
        """
        if task_type not in self._mappings:
            raise UnsupportedTaskSchemaError(
                f"Task schema type '{task_type}' not found in registry."
            )
        return self._mappings[task_type]
