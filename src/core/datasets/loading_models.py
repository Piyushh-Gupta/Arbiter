"""Immutable models for the Dataset Loading subsystem."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.core.datasets.mapping_models import TaskRecord
from src.core.datasets.serialization_models import DatasetManifest


class LoadingDefinition(BaseModel):
    """Base immutable configuration for a dataset loading target."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class ArbiterDataset(BaseModel):
    """Canonical immutable representation of a completely reconstructed dataset."""

    manifest: DatasetManifest
    records: tuple[TaskRecord, ...]

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
