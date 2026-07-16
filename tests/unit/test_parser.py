"""Unit tests for the M2.2 Dataset Parsing Layer."""

import io
import typing
from unittest.mock import MagicMock

import pytest

from src.core.datasets.loader_models import ArtifactHandle
from src.core.datasets.parser import DatasetParser
from src.core.datasets.parser_models import ParserConfig, ParserFormat
from src.core.datasets.parsers.csv_parser import CsvParser
from src.core.datasets.parsers.json_parser import JsonParser
from src.core.datasets.parsers.jsonl_parser import JsonlParser
from src.core.datasets.parsers.registry import ParserRegistry
from src.core.datasets.parsers.text_parser import PlainTextParser
from src.core.datasets.parsers.tsv_parser import TsvParser
from src.core.exceptions import ParserSyntaxError, UnsupportedFormatError


@pytest.fixture
def registry() -> ParserRegistry:
    reg = ParserRegistry()
    reg.register(ParserFormat.JSON, JsonParser())
    reg.register(ParserFormat.JSONL, JsonlParser())
    reg.register(ParserFormat.CSV, CsvParser())
    reg.register(ParserFormat.TSV, TsvParser())
    reg.register(ParserFormat.TXT, PlainTextParser())
    return reg


@pytest.fixture
def parser_service(registry: ParserRegistry) -> DatasetParser:
    return DatasetParser(registry)


def create_mock_handle(byte_data: bytes) -> MagicMock:
    stream = io.BytesIO(byte_data)
    handle = MagicMock(spec=ArtifactHandle)
    handle.__enter__.return_value = stream
    return handle


def test_registry_unsupported_format(parser_service: DatasetParser) -> None:
    """Test registry raises UnsupportedFormatError for unknown formats."""
    handle = create_mock_handle(b"")

    # We cheat Python's static typing temporarily to test runtime gates
    with pytest.raises(UnsupportedFormatError):
        list(parser_service.parse(handle, typing.cast(ParserFormat, "unknown_format")))


def test_json_parser_success(parser_service: DatasetParser) -> None:
    """Test standard JSON parsing yields records."""
    data = b'[{"id": 1}, {"id": 2}]'
    handle = create_mock_handle(data)

    records = list(parser_service.parse(handle, ParserFormat.JSON))
    assert len(records) == 2
    assert records[0].data == {"id": 1}
    assert records[1].data == {"id": 2}


def test_json_parser_malformed(parser_service: DatasetParser) -> None:
    """Test malformed JSON raises domain exception."""
    data = b'[{"id": 1'
    handle = create_mock_handle(data)

    with pytest.raises(ParserSyntaxError):
        list(parser_service.parse(handle, ParserFormat.JSON))


def test_jsonl_parser_success(parser_service: DatasetParser) -> None:
    """Test JSONL parsing yields lazy records."""
    data = b'{"id": 1}\n{"id": 2}\n'
    handle = create_mock_handle(data)

    records = list(parser_service.parse(handle, ParserFormat.JSONL))
    assert len(records) == 2
    assert records[0].data == {"id": 1}
    assert records[1].data == {"id": 2}


def test_csv_parser_success(parser_service: DatasetParser) -> None:
    """Test CSV parsing yields mapped dictionaries."""
    data = b"col1,col2\nval1,val2\nval3,val4\n"
    handle = create_mock_handle(data)

    records = list(parser_service.parse(handle, ParserFormat.CSV))
    assert len(records) == 2
    assert records[0].data == {"col1": "val1", "col2": "val2"}


def test_tsv_parser_success(parser_service: DatasetParser) -> None:
    """Test TSV parsing yields mapped dictionaries."""
    data = b"col1\tcol2\nval1\tval2\n"
    handle = create_mock_handle(data)

    records = list(parser_service.parse(handle, ParserFormat.TSV))
    assert len(records) == 1
    assert records[0].data == {"col1": "val1", "col2": "val2"}


def test_text_parser_success(parser_service: DatasetParser) -> None:
    """Test plain text parsing yields strings."""
    data = b"line 1\nline 2\n"
    handle = create_mock_handle(data)

    records = list(parser_service.parse(handle, ParserFormat.TXT))
    assert len(records) == 2
    assert records[0].data == "line 1"
    assert records[1].data == "line 2"


def test_custom_encoding() -> None:
    """Test config overrides encoding successfully."""
    # Data encoded in utf-16
    data = "test string\n".encode("utf-16")
    handle = create_mock_handle(data)

    # Configure parser manually instead of registry defaults
    config = ParserConfig(encoding="utf-16")
    parser = PlainTextParser(config)

    records = list(parser.parse(handle.__enter__()))
    assert len(records) == 1
    assert records[0].data == "test string"


def test_encoding_failure() -> None:
    """Test incorrect encoding raises syntactic error."""
    data = "test string".encode("utf-16")
    handle = create_mock_handle(data)

    # Defaults to utf-8, which should fail reading utf-16
    parser = PlainTextParser()

    with pytest.raises(ParserSyntaxError, match="Failed to decode"):
        list(parser.parse(handle.__enter__()))
