"""Pure evaluator for dataset validation."""

import json
import os
from collections.abc import Iterator
from pathlib import Path

import structlog

from src.core.datasets.artifact_manager import ArtifactManager
from src.core.datasets.artifact_models import ArtifactIdentity
from src.core.datasets.preprocessing_models import PreprocessedRecord
from src.core.datasets.validation_models import (
    ConstraintResult,
    FileConstraint,
    ManifestConstraint,
    ValidationConstraint,
    ValidationFailureCode,
    ValidationPipeline,
    ValidationReport,
)
from src.core.exceptions import DatasetValidationError, ValidationExecutionError

logger = structlog.get_logger(__name__)


class ArtifactValidator:
    """Read-only pure evaluator mapping a dataset directory state to a ValidationReport."""

    def __init__(self) -> None:
        pass

    def _evaluate_file_constraint(
        self, constraint: FileConstraint, dataset_dir: Path
    ) -> ConstraintResult:
        target_path = dataset_dir / constraint.target_path

        if not target_path.exists() or not target_path.is_file():
            return ConstraintResult(
                constraint_id=constraint.id,
                target_path=constraint.target_path,
                passed=False,
                failure_code=ValidationFailureCode.MISSING_FILE,
            )

        if not os.access(target_path, os.R_OK):
            return ConstraintResult(
                constraint_id=constraint.id,
                target_path=constraint.target_path,
                passed=False,
                failure_code=ValidationFailureCode.UNREADABLE,
            )

        if constraint.min_size_bytes is not None:
            if target_path.stat().st_size < constraint.min_size_bytes:
                return ConstraintResult(
                    constraint_id=constraint.id,
                    target_path=constraint.target_path,
                    passed=False,
                    failure_code=ValidationFailureCode.INSUFFICIENT_SIZE,
                )

        return ConstraintResult(
            constraint_id=constraint.id,
            target_path=constraint.target_path,
            passed=True,
        )

    def _evaluate_manifest_constraint(
        self, constraint: ManifestConstraint, dataset_dir: Path
    ) -> ConstraintResult:
        target_path = dataset_dir / constraint.target_path

        if not target_path.exists() or not target_path.is_file():
            return ConstraintResult(
                constraint_id=constraint.id,
                target_path=constraint.target_path,
                passed=False,
                failure_code=ValidationFailureCode.MISSING_FILE,
            )

        if not os.access(target_path, os.R_OK):
            return ConstraintResult(
                constraint_id=constraint.id,
                target_path=constraint.target_path,
                passed=False,
                failure_code=ValidationFailureCode.UNREADABLE,
            )

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                if constraint.is_jsonl:
                    # Validate JSONL by attempting to parse line by line
                    # We stream to avoid OOM on large manifests
                    for line in f:
                        if line.strip():
                            json.loads(line)
                else:
                    # We can use chunking/streaming parsers like ijson for huge JSONs,
                    # but for basic structure without external deps we use json.load
                    # If this is extremely large, memory error could occur, but standard json works for now.
                    json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, MemoryError) as e:
            logger.warning(
                "Manifest parsing failed", path=str(target_path), error=str(e)
            )
            return ConstraintResult(
                constraint_id=constraint.id,
                target_path=constraint.target_path,
                passed=False,
                failure_code=ValidationFailureCode.MANIFEST_CORRUPT,
            )

        return ConstraintResult(
            constraint_id=constraint.id,
            target_path=constraint.target_path,
            passed=True,
        )

    def validate(
        self,
        identity: ArtifactIdentity,
        constraints: tuple[ValidationConstraint, ...],
    ) -> ValidationReport:
        """
        Evaluate all constraints against the raw dataset directory.

        This method is strictly read-only and never raises domain exceptions upon validation failure.
        It returns a deterministic ValidationReport.
        """
        artifact = ArtifactManager.resolve_artifact(identity)
        dataset_dir = artifact.path

        results: list[ConstraintResult] = []
        is_valid = True

        logger.info("Starting dataset validation", canonical=identity.canonical)

        for constraint in constraints:
            if isinstance(constraint, FileConstraint):
                result = self._evaluate_file_constraint(constraint, dataset_dir)
            elif isinstance(constraint, ManifestConstraint):
                result = self._evaluate_manifest_constraint(constraint, dataset_dir)
            else:
                # Fallback for unknown constraint types to fail safe
                result = ConstraintResult(
                    constraint_id=constraint.id,
                    target_path=constraint.target_path,
                    passed=False,
                    # We map unknown to unreadable or missing, but structurally we just mark failed.
                    failure_code=ValidationFailureCode.MISSING_FILE,
                )

            results.append(result)
            if not result.passed:
                is_valid = False
                logger.warning(
                    "Constraint failed",
                    constraint_id=constraint.id,
                    failure_code=result.failure_code,
                )

        logger.info(
            "Completed dataset validation",
            canonical=identity.canonical,
            is_valid=is_valid,
        )

        return ValidationReport(
            dataset_id=identity.dataset_id,
            version=identity.version,
            is_valid=is_valid,
            results=tuple(results),
        )


class DatasetValidator:
    """Orchestrates deterministic dataset verification over immutable validation pipelines."""

    def validate(
        self, stream: Iterator[PreprocessedRecord], pipeline: ValidationPipeline
    ) -> Iterator[PreprocessedRecord]:
        """
        Executes the configured validation pipeline sequentially.

        Execution is purely declarative. The identical record object is yielded.
        Exceptions short-circuit the execution immediately.
        """
        try:
            for step in pipeline.steps:
                stream = step.strategy.validate_stream(
                    stream=stream, definition=step.definition
                )

            for record in stream:
                yield record

        except DatasetValidationError:
            # Expected domain errors propagate purely natively.
            raise
        except Exception as e:
            raise ValidationExecutionError(
                f"Unexpected failure during dataset validation execution: {e}"
            ) from e
