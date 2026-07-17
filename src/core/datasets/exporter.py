"""Orchestrator for executing dataset export pipelines."""

from src.core.datasets.export_models import ExportPipeline, SerializedArtifact


class DatasetExporter:
    """Orchestrator responsible for executing ExportPipelines."""

    def export(
        self,
        artifact: SerializedArtifact,
        pipeline: ExportPipeline,
    ) -> None:
        """
        Executes the provided serialized artifact against the pipeline sequentially.
        Fails fast if any step raises an ExportExecutionError.
        """
        for step in pipeline.steps:
            step.strategy.export(artifact, step.definition)
