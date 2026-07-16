"""TSV format parsing strategy."""

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


class TsvParser(BaseParser):
    """
    Parser strategy for TSV (Tab-Separated Values) files.
    """

    def __init__(self, config: ParserConfig | None = None) -> None:
        self.config = config or ParserConfig()

    def parse(self, stream: typing.BinaryIO) -> Iterator[ParsedRecord]:
        text_stream = io.TextIOWrapper(stream, encoding=self.config.encoding)

        try:
            reader = csv.DictReader(text_stream, delimiter="\t")
            for row in reader:
                yield ParsedRecord(data=typing.cast(dict[str, typing.Any], row))
        except csv.Error as e:
            raise ParserSyntaxError(f"Malformed TSV: {e}") from e
        except UnicodeDecodeError as e:
            raise ParserSyntaxError(
                f"Failed to decode TSV stream using encoding {self.config.encoding}: {e}"
            ) from e
