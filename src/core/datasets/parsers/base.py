"""Abstract base strategy for format-specific parsing."""

import typing
from collections.abc import Iterator

from src.core.datasets.parser_models import ParsedRecord


class BaseParser(typing.Protocol):
    """
    Abstract Strategy interface for parsing binary streams.

    Implementations must internally manage their own structural and decoding
    configurations and yield immutable ParsedRecord structures lazily.
    """

    def parse(self, stream: typing.BinaryIO) -> Iterator[ParsedRecord]:
        """
        Parses a binary stream iteratively, yielding syntactic records.

        Args:
            stream: A read-only binary stream.

        Yields:
            ParsedRecord instances containing strictly typed syntactic data.

        Raises:
            ParserSyntaxError: If the stream contains malformed or unparseable data.
        """
        ...
