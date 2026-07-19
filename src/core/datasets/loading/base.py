"""Stateless base loader protocol."""

from __future__ import annotations

import typing

from src.core.datasets.export_models import SerializedArtifact
from src.core.datasets.loading_models import ArbiterDataset, LoadingDefinition


@typing.runtime_checkable
class BaseLoader(typing.Protocol):
    """Stateless protocol for all single-target loading strategies."""

    def validate_compatibility(self, definition: LoadingDefinition) -> None:
        """Statically verifies if this loader supports the given definition."""
        ...

    def load(
        self, artifact: SerializedArtifact, definition: LoadingDefinition
    ) -> ArbiterDataset:
        """
        Executes the dataset reconstruction.

        Receives:
        - artifact: The unified, immutable physical location/handle representing the dataset.
        - definition: The validated immutable configuration parameters.

        Returns:
        - ArbiterDataset: A fully materialized, immutable dataset object containing the parsed records and manifest.
        """
        ...
