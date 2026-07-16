"""Registry for dataset parsing strategies."""

from src.core.datasets.parser_models import ParserFormat
from src.core.datasets.parsers.base import BaseParser
from src.core.exceptions import UnsupportedFormatError


class ParserRegistry:
    """
    Registry maintaining mappings between ParserFormat and BaseParser instances.
    Follows an immutable configuration pattern once populated.
    """

    def __init__(self) -> None:
        self._parsers: dict[ParserFormat, BaseParser] = {}

    def register(self, format_type: ParserFormat, parser: BaseParser) -> None:
        """
        Register a concrete parser strategy for a specific format.

        Args:
            format_type: The strongly typed format enum.
            parser: The configured parser instance.
        """
        self._parsers[format_type] = parser

    def resolve(self, format_type: ParserFormat) -> BaseParser:
        """
        Resolve the appropriate parser strategy for the given format.

        Args:
            format_type: The strongly typed format enum to resolve.

        Returns:
            The registered BaseParser instance.

        Raises:
            UnsupportedFormatError: If the format is not registered.
        """
        if format_type not in self._parsers:
            # We use getattr in case a raw string sneaks past typing at runtime
            val = getattr(format_type, "value", format_type)
            raise UnsupportedFormatError(
                f"No parser strategy registered for format: {val}"
            )
        return self._parsers[format_type]
