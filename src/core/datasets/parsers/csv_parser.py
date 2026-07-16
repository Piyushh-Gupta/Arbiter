"""CSV format parsing strategy."""

import csv
import io
import sys
import typing
from collections.abc import Iterator

from src.core.datasets.parser_models import ParsedRecord, ParserConfig
from src.core.datasets.parsers.base import BaseParser
from src.core.exceptions import ParserSyntaxError

# Mitigate CSV memory limits proactively and safely
_max_size = sys.maxsize
while True:
    try:
        csv.field_size_limit(_max_size)
        break
    except OverflowError:
        _max_size = int(_max_size / 10)


class CsvParser(BaseParser):
    """
    Parser strategy for standard CSV files.
    """

    def __init__(self, config: ParserConfig | None = None) -> None:
        self.config = config or ParserConfig()

    def parse(self, stream: typing.BinaryIO) -> Iterator[ParsedRecord]:
        text_stream = io.TextIOWrapper(stream, encoding=self.config.encoding)

        try:
            reader = csv.DictReader(text_stream)
            for row_num, row in enumerate(
                reader, start=2
            ):  # start=2 considering header is line 1
                # DictReader yields dicts mapping header names to string values
                # If a row is missing headers or values, csv modules handle it, but we can catch explicit errors.
                yield ParsedRecord(data=typing.cast(dict[str, typing.Any], row))
        except csv.Error as e:
            raise ParserSyntaxError(f"Malformed CSV: {e}") from e
        except UnicodeDecodeError as e:
            raise ParserSyntaxError(
                f"Failed to decode CSV stream using encoding {self.config.encoding}: {e}"
            ) from e
