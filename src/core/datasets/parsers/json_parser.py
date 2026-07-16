"""JSON format parsing strategy."""

import io
import json
import typing
from collections.abc import Iterator

from src.core.datasets.parser_models import ParsedRecord, ParserConfig
from src.core.datasets.parsers.base import BaseParser
from src.core.exceptions import ParserSyntaxError


class JsonParser(BaseParser):
    """
    Parser strategy for standard JSON files.
    """

    def __init__(self, config: ParserConfig | None = None) -> None:
        self.config = config or ParserConfig()

    def parse(self, stream: typing.BinaryIO) -> Iterator[ParsedRecord]:
        text_stream = io.TextIOWrapper(stream, encoding=self.config.encoding)

        try:
            data = json.load(text_stream)
        except json.JSONDecodeError as e:
            raise ParserSyntaxError(f"Malformed JSON: {e}") from e

        # If it's a list, we lazily yield each item.
        # Note: Standard `json.load` eagerly constructs the AST in memory.
        # True constant-memory streaming for massive monolithic JSON arrays
        # requires a 3rd party library like `ijson`.
        if isinstance(data, list):
            for item in data:
                yield ParsedRecord(data=item)
        else:
            yield ParsedRecord(data=data)
