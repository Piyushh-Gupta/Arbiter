"""JSON Lines (JSONL) format parsing strategy."""

import io
import json
import typing
from collections.abc import Iterator

from src.core.datasets.parser_models import ParsedRecord, ParserConfig
from src.core.datasets.parsers.base import BaseParser
from src.core.exceptions import ParserSyntaxError


class JsonlParser(BaseParser):
    """
    Parser strategy for JSON Lines files.
    """

    def __init__(self, config: ParserConfig | None = None) -> None:
        self.config = config or ParserConfig()

    def parse(self, stream: typing.BinaryIO) -> Iterator[ParsedRecord]:
        text_stream = io.TextIOWrapper(stream, encoding=self.config.encoding)

        for line_num, line in enumerate(text_stream, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ParserSyntaxError(
                    f"Malformed JSONL at line {line_num}: {e}"
                ) from e

            yield ParsedRecord(data=data)
