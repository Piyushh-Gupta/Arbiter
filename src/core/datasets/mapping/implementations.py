"""Task-specific mapper implementations."""

from collections.abc import Iterator
from typing import Any

from src.core.datasets.mapping_models import (
    ClassificationRecord,
    FactVerificationRecord,
    QARecord,
    SchemaDefinition,
)
from src.core.datasets.normalization_models import NormalizedRecord
from src.core.exceptions import MissingRequiredFieldError


def _extract_field(
    content: dict[str, Any] | str, source_field: str | None, required: bool = True
) -> Any:
    """Extracts a field from canonical content based on the source field mapping."""
    if not source_field:
        if required:
            raise MissingRequiredFieldError("Required field mapping is empty.")
        return None

    if not isinstance(content, dict):
        if required:
            raise MissingRequiredFieldError(
                f"Cannot extract '{source_field}' from non-dictionary content."
            )
        return None

    if source_field not in content:
        if required:
            raise MissingRequiredFieldError(
                f"Required field '{source_field}' missing from content."
            )
        return None

    return content[source_field]


class FactVerificationMapper:
    """Task mapper for Fact Verification / NLI."""

    def map_stream(
        self, stream: Iterator[NormalizedRecord], schema_def: SchemaDefinition
    ) -> Iterator[FactVerificationRecord]:
        """Maps canonical records to FactVerificationRecord."""
        claim_src = schema_def.field_map.get("claim")
        evidence_src = schema_def.field_map.get("evidence")
        label_src = schema_def.field_map.get("label")

        for record in stream:
            claim_val = str(_extract_field(record.content, claim_src, required=True))

            # Optional fields
            evidence_val = None
            if (
                evidence_src
                and isinstance(record.content, dict)
                and evidence_src in record.content
            ):
                evidence_val = str(record.content[evidence_src])

            label_val = None
            if (
                label_src
                and isinstance(record.content, dict)
                and label_src in record.content
            ):
                label_val = str(record.content[label_src])

            yield FactVerificationRecord(
                claim=claim_val,
                evidence=evidence_val,
                label=label_val,
                provenance=record.provenance,
            )


class QAMapper:
    """Task mapper for Question Answering."""

    def map_stream(
        self, stream: Iterator[NormalizedRecord], schema_def: SchemaDefinition
    ) -> Iterator[QARecord]:
        """Maps canonical records to QARecord."""
        question_src = schema_def.field_map.get("question")
        context_src = schema_def.field_map.get("context")
        answer_src = schema_def.field_map.get("answer")

        for record in stream:
            question_val = str(
                _extract_field(record.content, question_src, required=True)
            )

            context_val = None
            if (
                context_src
                and isinstance(record.content, dict)
                and context_src in record.content
            ):
                context_val = str(record.content[context_src])

            answer_val = None
            if (
                answer_src
                and isinstance(record.content, dict)
                and answer_src in record.content
            ):
                answer_val = str(record.content[answer_src])

            yield QARecord(
                question=question_val,
                context=context_val,
                answer=answer_val,
                provenance=record.provenance,
            )


class ClassificationMapper:
    """Task mapper for Text Classification."""

    def map_stream(
        self, stream: Iterator[NormalizedRecord], schema_def: SchemaDefinition
    ) -> Iterator[ClassificationRecord]:
        """Maps canonical records to ClassificationRecord."""
        text_src = schema_def.field_map.get("text")
        label_src = schema_def.field_map.get("label")

        for record in stream:
            text_val = str(_extract_field(record.content, text_src, required=True))

            label_val = None
            if (
                label_src
                and isinstance(record.content, dict)
                and label_src in record.content
            ):
                label_val = str(record.content[label_src])

            yield ClassificationRecord(
                text=text_val,
                label=label_val,
                provenance=record.provenance,
            )
