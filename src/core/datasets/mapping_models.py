"""Data models for the Schema Mapping Layer (M2.4)."""

from enum import Enum

from pydantic import BaseModel, ConfigDict

from src.core.datasets.normalization_models import ProvenanceMetadata


class TaskSchemaType(str, Enum):
    """Enumeration of supported AI task schema types."""

    FACT_VERIFICATION = "fact_verification"
    QA = "qa"
    CLASSIFICATION = "classification"


class SchemaDefinition(BaseModel):
    """
    Immutable model representing target-to-source semantic mapping configurations.

    This acts as the declarative configuration, strictly decoupling field extraction
    knowledge from the execution mapper implementations.
    """

    task_type: TaskSchemaType
    field_map: dict[str, str]

    model_config = ConfigDict(frozen=True)


class TaskRecord(BaseModel):
    """
    Abstract canonical base contract establishing foundational downstream infrastructure tracking.

    Every specific downstream task record inherits from this base to ensure infrastructure
    metadata (e.g. provenance) is strictly carried forward.
    """

    provenance: ProvenanceMetadata

    model_config = ConfigDict(frozen=True)


class FactVerificationRecord(TaskRecord):
    """Immutable downstream canonical model representing NLI/Fact Verification tasks."""

    claim: str
    evidence: str | None
    label: str | None


class QARecord(TaskRecord):
    """Immutable downstream canonical model representing Question Answering tasks."""

    question: str
    context: str | None
    answer: str | None


class ClassificationRecord(TaskRecord):
    """Immutable downstream canonical model representing text classification tasks."""

    text: str
    label: str | None
