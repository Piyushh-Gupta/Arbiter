"""Unit tests for the Dataset Export Framework."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import BotoCoreError
from huggingface_hub.utils import HfHubHTTPError  # type: ignore[attr-defined]
from pydantic import ConfigDict, ValidationError

from src.core.datasets.export.base import BaseExporter
from src.core.datasets.export.implementations import (
    HuggingFaceExporter,
    LocalExporter,
    S3Exporter,
)
from src.core.datasets.export_models import (
    ExportDefinition,
    ExportPipeline,
    ExportProfile,
    ExportProfileRegistry,
    ExportStep,
    HuggingFaceExportDefinition,
    HuggingFaceRepositoryType,
    LocalExportDefinition,
    S3ExportDefinition,
    S3StorageClass,
    SerializedArtifact,
)
from src.core.datasets.exporter import DatasetExporter
from src.core.exceptions import (
    DuplicateExportProfileError,
    ExportConfigurationError,
    ExportExecutionError,
    ExportProfileNotFoundError,
)


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


def test_local_export_definition_immutability() -> None:
    """Ensure LocalExportDefinition is frozen."""
    definition = LocalExportDefinition(destination_root=Path("/dest"))
    with pytest.raises(ValidationError):
        definition.destination_root = Path("/mutated")


def test_local_exporter_compatibility() -> None:
    """Ensure LocalExporter rejects invalid definitions."""
    exporter = LocalExporter()
    definition = MockExportDefinition(target_name="test")
    with pytest.raises(
        ExportConfigurationError, match="LocalExporter requires LocalExportDefinition"
    ):
        exporter.validate_compatibility(definition)


def test_local_exporter_file_export() -> None:
    """Ensure exact single file export and destination parent creation."""
    exporter = LocalExporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        source_file = root_path / "source.txt"
        source_file.write_text("hello world")

        artifact = SerializedArtifact(root_path=source_file)

        dest_file = root_path / "nested" / "dest" / "target.txt"
        definition = LocalExportDefinition(destination_root=dest_file)

        exporter.export(artifact, definition)

        assert dest_file.exists()
        assert dest_file.read_text() == "hello world"


def test_local_exporter_dir_export() -> None:
    """Ensure exact directory export."""
    exporter = LocalExporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        source_dir = root_path / "source_dir"
        source_dir.mkdir()
        (source_dir / "file1.txt").write_text("data1")
        (source_dir / "file2.txt").write_text("data2")

        artifact = SerializedArtifact(root_path=source_dir)

        dest_dir = root_path / "dest_dir"
        definition = LocalExportDefinition(destination_root=dest_dir)

        exporter.export(artifact, definition)

        assert dest_dir.exists()
        assert dest_dir.is_dir()
        assert (dest_dir / "file1.txt").read_text() == "data1"
        assert (dest_dir / "file2.txt").read_text() == "data2"


def test_local_exporter_overwrite_disabled() -> None:
    """Ensure export fails when destination exists and overwrite is disabled."""
    exporter = LocalExporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        source_file = root_path / "source.txt"
        source_file.write_text("hello")
        artifact = SerializedArtifact(root_path=source_file)

        dest_file = root_path / "dest.txt"
        dest_file.write_text("existing")
        definition = LocalExportDefinition(
            destination_root=dest_file, overwrite_existing=False
        )

        with pytest.raises(
            ExportExecutionError, match="already exists and overwrite is disabled"
        ):
            exporter.export(artifact, definition)


def test_local_exporter_overwrite_enabled_clears_stale_files() -> None:
    """Ensure overwrite=True completely replaces the target, leaving no stale files."""
    exporter = LocalExporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)

        # Source with one file
        source_dir = root_path / "source_dir"
        source_dir.mkdir()
        (source_dir / "new_file.txt").write_text("new_data")
        artifact = SerializedArtifact(root_path=source_dir)

        # Dest with a stale file
        dest_dir = root_path / "dest_dir"
        dest_dir.mkdir()
        (dest_dir / "stale_file.txt").write_text("stale_data")
        definition = LocalExportDefinition(
            destination_root=dest_dir, overwrite_existing=True
        )

        exporter.export(artifact, definition)

        assert dest_dir.exists()
        assert (dest_dir / "new_file.txt").exists()
        assert not (dest_dir / "stale_file.txt").exists()


def test_local_exporter_os_error_wrapping() -> None:
    """Ensure OS errors are safely wrapped."""
    exporter = LocalExporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        # Source doesn't exist, will cause FileNotFoundError in shutil
        source_file = root_path / "missing.txt"
        artifact = SerializedArtifact(root_path=source_file)

        dest_file = root_path / "dest.txt"
        definition = LocalExportDefinition(destination_root=dest_file)

        with pytest.raises(ExportExecutionError, match="Failed to export"):
            exporter.export(artifact, definition)


def test_huggingface_export_definition_immutability() -> None:
    """Ensure HuggingFaceExportDefinition is frozen and enum validation works."""
    definition = HuggingFaceExportDefinition(
        repository_id="test/repo",
        repository_type=HuggingFaceRepositoryType.DATASET,
    )
    with pytest.raises(ValidationError):
        definition.repository_id = "test/mutated"

    with pytest.raises(ValidationError):
        HuggingFaceExportDefinition(
            repository_id="test/repo",
            repository_type="invalid_type",  # type: ignore[arg-type]
        )


def test_huggingface_exporter_compatibility() -> None:
    """Ensure HuggingFaceExporter rejects invalid definitions."""
    exporter = HuggingFaceExporter()
    definition = MockExportDefinition(target_name="test")
    with pytest.raises(
        ExportConfigurationError,
        match="HuggingFaceExporter requires HuggingFaceExportDefinition",
    ):
        exporter.validate_compatibility(definition)


@patch("src.core.datasets.export.implementations.HfApi")
def test_huggingface_exporter_file_upload(mock_hf_api_cls: MagicMock) -> None:
    """Ensure exact single file upload."""
    mock_api = mock_hf_api_cls.return_value
    exporter = HuggingFaceExporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        source_file = root_path / "source.txt"
        source_file.write_text("hello world")

        artifact = SerializedArtifact(root_path=source_file)
        definition = HuggingFaceExportDefinition(
            repository_id="test/repo",
            create_repo_if_missing=True,
        )

        exporter.export(artifact, definition)

        mock_api.create_repo.assert_called_once_with(
            repo_id="test/repo",
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        mock_api.upload_file.assert_called_once_with(
            path_or_fileobj=str(source_file),
            path_in_repo="source.txt",
            repo_id="test/repo",
            repo_type="dataset",
            revision="main",
            commit_message="Upload dataset via Arbiter Export Subsystem",
        )


@patch("src.core.datasets.export.implementations.HfApi")
def test_huggingface_exporter_dir_upload(mock_hf_api_cls: MagicMock) -> None:
    """Ensure exact directory upload."""
    mock_api = mock_hf_api_cls.return_value
    exporter = HuggingFaceExporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        source_dir = root_path / "source_dir"
        source_dir.mkdir()

        artifact = SerializedArtifact(root_path=source_dir)
        definition = HuggingFaceExportDefinition(
            repository_id="test/repo",
            create_repo_if_missing=True,
        )

        exporter.export(artifact, definition)

        mock_api.create_repo.assert_called_once_with(
            repo_id="test/repo",
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        mock_api.upload_folder.assert_called_once_with(
            folder_path=str(source_dir),
            repo_id="test/repo",
            repo_type="dataset",
            revision="main",
            commit_message="Upload dataset via Arbiter Export Subsystem",
        )


@patch("src.core.datasets.export.implementations.HfApi")
def test_huggingface_exporter_repo_missing_error(mock_hf_api_cls: MagicMock) -> None:
    """Ensure create_repo_if_missing=False checks dataset_info and raises on fail."""
    mock_api = mock_hf_api_cls.return_value
    mock_api.dataset_info.side_effect = Exception("Repository Not Found")

    exporter = HuggingFaceExporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        source_dir = root_path / "source_dir"
        source_dir.mkdir()

        artifact = SerializedArtifact(root_path=source_dir)
        definition = HuggingFaceExportDefinition(
            repository_id="test/repo",
            create_repo_if_missing=False,
        )

        with pytest.raises(
            ExportExecutionError, match="Repository test/repo does not exist"
        ):
            exporter.export(artifact, definition)

        mock_api.create_repo.assert_not_called()
        mock_api.upload_folder.assert_not_called()


@patch("src.core.datasets.export.implementations.HfApi")
def test_huggingface_exporter_http_error_wrapping(mock_hf_api_cls: MagicMock) -> None:
    """Ensure HfHubHTTPError is wrapped."""
    mock_api = mock_hf_api_cls.return_value
    response_mock = MagicMock()
    response_mock.status_code = 500
    mock_api.upload_folder.side_effect = HfHubHTTPError(
        "Server error", response=response_mock
    )

    exporter = HuggingFaceExporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        source_dir = root_path / "source_dir"
        source_dir.mkdir()

        artifact = SerializedArtifact(root_path=source_dir)
        definition = HuggingFaceExportDefinition(repository_id="test/repo")

        with pytest.raises(ExportExecutionError, match="Hugging Face HTTP error"):
            exporter.export(artifact, definition)


def test_s3_export_definition_immutability() -> None:
    """Ensure S3ExportDefinition is frozen and enum validation works."""
    definition = S3ExportDefinition(
        bucket_name="test-bucket",
        object_prefix="test/prefix",
        storage_class=S3StorageClass.STANDARD,
    )
    with pytest.raises(ValidationError):
        definition.bucket_name = "test-mutated"

    with pytest.raises(ValidationError):
        S3ExportDefinition(
            bucket_name="test-bucket",
            object_prefix="test/prefix",
            storage_class="INVALID_CLASS",  # type: ignore[arg-type]
        )


def test_s3_exporter_compatibility() -> None:
    """Ensure S3Exporter rejects invalid definitions."""
    exporter = S3Exporter()
    definition = MockExportDefinition(target_name="test")
    with pytest.raises(
        ExportConfigurationError, match="S3Exporter requires S3ExportDefinition"
    ):
        exporter.validate_compatibility(definition)


@patch("src.core.datasets.export.implementations.boto3")
def test_s3_exporter_file_upload(mock_boto3: MagicMock) -> None:
    """Ensure exact single file upload to S3."""
    mock_s3_client = mock_boto3.client.return_value
    exporter = S3Exporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        source_file = root_path / "source.txt"
        source_file.write_text("hello world")

        artifact = SerializedArtifact(root_path=source_file)
        definition = S3ExportDefinition(
            bucket_name="test-bucket",
            object_prefix="my/prefix",
            storage_class=S3StorageClass.STANDARD_IA,
            region_name="us-east-1",
        )

        exporter.export(artifact, definition)

        mock_boto3.client.assert_called_once_with("s3", region_name="us-east-1")
        mock_s3_client.upload_file.assert_called_once_with(
            Filename=str(source_file),
            Bucket="test-bucket",
            Key="my/prefix/source.txt",
            ExtraArgs={"StorageClass": "STANDARD_IA"},
        )


@patch("src.core.datasets.export.implementations.boto3")
def test_s3_exporter_dir_upload(mock_boto3: MagicMock) -> None:
    """Ensure exact directory upload and normalization of keys."""
    mock_s3_client = mock_boto3.client.return_value
    exporter = S3Exporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        source_dir = root_path / "source_dir"
        source_dir.mkdir()
        (source_dir / "file1.txt").write_text("data1")
        nested_dir = source_dir / "nested"
        nested_dir.mkdir()
        (nested_dir / "file2.txt").write_text("data2")

        artifact = SerializedArtifact(root_path=source_dir)
        definition = S3ExportDefinition(
            bucket_name="test-bucket",
            object_prefix="my/prefix",
        )

        exporter.export(artifact, definition)

        assert mock_s3_client.upload_file.call_count == 2

        calls = mock_s3_client.upload_file.call_args_list
        # Keys should be normalized to / regardless of OS (Windows/Linux)
        # We can just check the keys exist in the calls
        keys = [call.kwargs["Key"] for call in calls]

        assert "my/prefix/file1.txt" in keys
        assert "my/prefix/nested/file2.txt" in keys


@patch("src.core.datasets.export.implementations.boto3")
def test_s3_exporter_botocore_error_wrapping(mock_boto3: MagicMock) -> None:
    """Ensure BotoCoreError is wrapped."""
    mock_s3_client = mock_boto3.client.return_value
    mock_s3_client.upload_file.side_effect = BotoCoreError()

    exporter = S3Exporter()

    with tempfile.TemporaryDirectory() as tmpdir:
        root_path = Path(tmpdir)
        source_file = root_path / "source.txt"
        source_file.write_text("test")

        artifact = SerializedArtifact(root_path=source_file)
        definition = S3ExportDefinition(bucket_name="test-bucket", object_prefix="pre")

        with pytest.raises(ExportExecutionError, match="Amazon S3 export failed"):
            exporter.export(artifact, definition)


def test_export_profile_immutability() -> None:
    """Ensure ExportProfile is immutable."""
    profile = ExportProfile(profile_id="test", pipeline=ExportPipeline(steps=()))
    with pytest.raises(ValidationError):
        profile.profile_id = "mutated"


def test_export_profile_registry_construction() -> None:
    """Ensure ExportProfileRegistry rejects duplicates eagerly."""
    p1 = ExportProfile(profile_id="test", pipeline=ExportPipeline(steps=()))
    p2 = ExportProfile(profile_id="test", pipeline=ExportPipeline(steps=()))

    with pytest.raises(DuplicateExportProfileError, match="is already registered"):
        ExportProfileRegistry(profiles=(p1, p2))


def test_export_profile_registry_resolution() -> None:
    """Ensure registry resolves profiles accurately and rejects unknowns."""
    p1 = ExportProfile(profile_id="p1", pipeline=ExportPipeline(steps=()))
    registry = ExportProfileRegistry(profiles=(p1,))

    # Successful resolution
    resolved = registry.resolve("p1")
    assert resolved is p1

    # Unknown resolution
    with pytest.raises(ExportProfileNotFoundError, match="not found in registry"):
        registry.resolve("unknown")


def test_export_profile_execution_equivalence() -> None:
    """Ensure extracting pipeline from profile yields identical execution behavior."""
    mock_exporter_1 = MagicMock(spec=BaseExporter)
    mock_exporter_2 = MagicMock(spec=BaseExporter)

    # Exporter must return None from validate_compatibility to pass
    mock_exporter_1.validate_compatibility.return_value = None
    mock_exporter_2.validate_compatibility.return_value = None

    artifact = SerializedArtifact(root_path=Path("/tmp/fake"))

    step1 = ExportStep(
        definition=MockExportDefinition(target_name="dest1"), strategy=mock_exporter_1
    )
    step2 = ExportStep(
        definition=MockExportDefinition(target_name="dest2"), strategy=mock_exporter_2
    )
    pipeline = ExportPipeline(steps=(step1, step2))

    # Setup profile
    profile = ExportProfile(profile_id="prod", pipeline=pipeline)
    registry = ExportProfileRegistry(profiles=(profile,))

    # Scenario 1: Direct Execution
    exporter_orchestrator = DatasetExporter()
    exporter_orchestrator.export(artifact, pipeline)

    call_args_1_direct = mock_exporter_1.export.call_args[0]
    call_args_2_direct = mock_exporter_2.export.call_args[0]

    assert call_args_1_direct[0] is artifact
    assert call_args_2_direct[0] is artifact

    # Reset mocks
    mock_exporter_1.reset_mock()
    mock_exporter_2.reset_mock()

    # Scenario 2: Profile Execution
    resolved_profile = registry.resolve("prod")
    exporter_orchestrator.export(artifact, resolved_profile.pipeline)

    call_args_1_profile = mock_exporter_1.export.call_args[0]
    call_args_2_profile = mock_exporter_2.export.call_args[0]

    # Verify execution identity
    assert call_args_1_profile[0] is artifact
    assert call_args_2_profile[0] is artifact
    assert mock_exporter_1.export.call_count == 1
    assert mock_exporter_2.export.call_count == 1
