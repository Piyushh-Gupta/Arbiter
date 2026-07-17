"""Unit tests for the Dataset Export Framework."""

import tempfile
from pathlib import Path

import pytest
from pydantic import ConfigDict, ValidationError

from src.core.datasets.export.base import BaseExporter
from src.core.datasets.export_models import (
    ExportDefinition,
    ExportPipeline,
    ExportStep,
    SerializedArtifact,
)
from src.core.datasets.exporter import DatasetExporter
from src.core.exceptions import ExportConfigurationError, ExportExecutionError


class MockExportDefinition(ExportDefinition):
    """A mock configuration for testing."""

    target_name: str
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class MockExporter(BaseExporter):
    """A mock strategy that appends to a trace list on export."""

    def __init__(self, trace_list: list[str]) -> None:
        self.trace_list = trace_list

    def export(
        self, artifact: SerializedArtifact, definition: ExportDefinition
    ) -> None:
        assert isinstance(definition, MockExportDefinition)
        self.trace_list.append(
            f"export:{definition.target_name}:{artifact.root_path.name}"
        )

    def validate_compatibility(self, definition: ExportDefinition) -> None:
        if not isinstance(definition, MockExportDefinition):
            raise ExportConfigurationError("Incompatible definition type.")


class BadMockExporter(BaseExporter):
    """A mock strategy that always fails compatibility validation."""

    def export(
        self, artifact: SerializedArtifact, definition: ExportDefinition
    ) -> None:
        pass

    def validate_compatibility(self, definition: ExportDefinition) -> None:
        raise ExportConfigurationError("Always fails.")


class ErrorMockExporter(BaseExporter):
    """A mock strategy that raises an execution error."""

    def export(
        self, artifact: SerializedArtifact, definition: ExportDefinition
    ) -> None:
        raise ExportExecutionError("Network failure.")

    def validate_compatibility(self, definition: ExportDefinition) -> None:
        pass


def test_serialized_artifact_immutability() -> None:
    """Ensure SerializedArtifact is frozen."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = SerializedArtifact(root_path=Path(tmpdir))
        with pytest.raises(ValidationError):
            artifact.root_path = Path("/mutated")


def test_export_definition_immutability() -> None:
    """Ensure ExportDefinition is frozen."""
    definition = MockExportDefinition(target_name="test")
    with pytest.raises(ValidationError):
        definition.target_name = "mutated"


def test_export_pipeline_immutability() -> None:
    """Ensure ExportPipeline is frozen."""
    definition = MockExportDefinition(target_name="test")
    strategy = MockExporter(trace_list=[])
    step = ExportStep(definition=definition, strategy=strategy)
    pipeline = ExportPipeline(steps=(step,))

    with pytest.raises(ValidationError):
        pipeline.steps = ()


def test_export_step_compatibility_validation() -> None:
    """Ensure ExportStep fails fast if strategy is incompatible with definition."""
    definition = MockExportDefinition(target_name="test")
    strategy = BadMockExporter()

    with pytest.raises(ExportConfigurationError, match="Always fails."):
        ExportStep(definition=definition, strategy=strategy)


def test_dataset_exporter_orchestration() -> None:
    """Ensure DatasetExporter orchestrates sequentially and statelessly."""
    trace: list[str] = []

    step1 = ExportStep(
        definition=MockExportDefinition(target_name="A"),
        strategy=MockExporter(trace),
    )
    step2 = ExportStep(
        definition=MockExportDefinition(target_name="B"),
        strategy=MockExporter(trace),
    )

    pipeline = ExportPipeline(steps=(step1, step2))
    orchestrator = DatasetExporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        artifact = SerializedArtifact(root_path=root_path)

        # Execute once
        orchestrator.export(artifact, pipeline)

        # Execute twice to ensure stateless execution
        orchestrator.export(artifact, pipeline)

        # Verify exact sequence
        expected_name = root_path.name
        assert trace == [
            f"export:A:{expected_name}",
            f"export:B:{expected_name}",
            f"export:A:{expected_name}",
            f"export:B:{expected_name}",
        ]


def test_dataset_exporter_fail_fast() -> None:
    """Ensure DatasetExporter halts execution immediately on failure."""
    trace: list[str] = []

    step1 = ExportStep(
        definition=MockExportDefinition(target_name="A"),
        strategy=MockExporter(trace),
    )
    step_error = ExportStep(
        definition=MockExportDefinition(target_name="ERR"),
        strategy=ErrorMockExporter(),
    )
    step3 = ExportStep(
        definition=MockExportDefinition(target_name="C"),
        strategy=MockExporter(trace),
    )

    pipeline = ExportPipeline(steps=(step1, step_error, step3))
    orchestrator = DatasetExporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        artifact = SerializedArtifact(root_path=root_path)

        with pytest.raises(ExportExecutionError, match="Network failure."):
            orchestrator.export(artifact, pipeline)

        expected_name = root_path.name

        # Verify step1 executed, but step3 did not
        assert trace == [f"export:A:{expected_name}"]
