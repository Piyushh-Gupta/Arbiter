"""Orchestrator for executing dataset serialization pipelines."""

from typing import Iterator

from src.core.datasets.preprocessing_models import PreprocessedRecord
from src.core.datasets.serialization_models import SerializationPipeline


class DatasetSerializer:
    """Orchestrator responsible for executing SerializationPipelines."""

    def serialize(
        self,
        stream: Iterator[PreprocessedRecord],
        pipeline: SerializationPipeline,
    ) -> None:
        """
        Executes the provided stream against the provided pipeline in a single pass.

        Fully consumes the iterator to drive the serialization execution.
        Assumes the pipeline has already been validated during construction.
        """
        # Assemble execution chain
        current_stream = stream
        for step in pipeline.steps:
            current_stream = step.strategy.serialize_stream(
                current_stream, step.definition
            )

        # Exhaust the iterator to strictly drive serialization side-effects
        for _ in current_stream:
            pass
