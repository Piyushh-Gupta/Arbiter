"""Immutable models defining the output boundaries and formats of the parsing layer."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, JsonValue


class ParserFormat(str, Enum):
    """Strongly typed file formats natively supported by the parsing layer."""

    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    TSV = "tsv"
    TXT = "txt"


class ParserConfig(BaseModel):
    """
    Configuration governing syntactic parsing and decoding policy.
    Owned internally by format-specific parser strategies.
    """

    encoding: str = "utf-8"

    model_config = ConfigDict(frozen=True)


ParsedData = dict[str, JsonValue] | str


class ParsedRecord(BaseModel):
    """
    Immutable syntactic record representing one unit of extracted data.
    Explicitly NOT normalized, validated, dataset-aware, or feature engineered.
    """

    data: ParsedData

    model_config = ConfigDict(frozen=True)
