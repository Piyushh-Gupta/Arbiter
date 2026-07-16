"""Plain text format parsing strategy."""

import io
import typing
from collections.abc import Iterator

from src.core.datasets.parser_models import ParsedRecord, ParserConfig
from src.core.datasets.parsers.base import BaseParser
from src.core.exceptions import ParserSyntaxError


class PlainTextParser(BaseParser):
    """
    Parser strategy for generic plain text files.
    Lazily yields each line as a distinct ParsedRecord.
    """

    def __init__(self, config: ParserConfig | None = None) -> None:
        self.config = config or ParserConfig()

    def parse(self, stream: typing.BinaryIO) -> Iterator[ParsedRecord]:
        text_stream = io.TextIOWrapper(stream, encoding=self.config.encoding)

        try:
            for line in text_stream:
                # We yield the raw string, keeping \n or stripping it depending on exact specs.
                # Usually it's better to strip trailing newlines for plain text records.
                yield ParsedRecord(data=line.strip("\n\r"))
        except UnicodeDecodeError as e:
            raise ParserSyntaxError(
                f"Failed to decode text stream using encoding {self.config.encoding}: {e}"
            ) from e
