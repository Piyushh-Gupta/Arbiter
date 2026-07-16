"""Immutable models establishing Arbiter's internal data contract."""

from pydantic import BaseModel, ConfigDict, JsonValue


class ProvenanceMetadata(BaseModel):
    """
    Lightweight infrastructure tracking metadata.
    This represents system-level observability, strictly avoiding dataset semantic features.
    """

    record_index: int

    model_config = ConfigDict(frozen=True)


class NormalizedRecord(BaseModel):
    """
    Immutable canonical record representing the singular output standard for all datasets.
    Establishes the definitive contract for M2.4 and beyond.
    """

    content: dict[str, JsonValue] | str
    provenance: ProvenanceMetadata

    model_config = ConfigDict(frozen=True)
