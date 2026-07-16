"""DatasetParser orchestration layer."""

from collections.abc import Iterator

from src.core.datasets.loader_models import ArtifactHandle
from src.core.datasets.parser_models import ParsedRecord, ParserFormat
from src.core.datasets.parsers.registry import ParserRegistry


class DatasetParser:
    """
    Instance-based orchestration service for format-agnostic parsing.
    Requires an injected ParserRegistry to resolve strategies dynamically.
    """

    def __init__(self, registry: ParserRegistry) -> None:
        self._registry = registry

    def parse(
        self, handle: ArtifactHandle, format_type: ParserFormat
    ) -> Iterator[ParsedRecord]:
        """
        Delegates the ArtifactHandle stream to the correct formatting strategy via the registry.
        The requested parser is explicitly defined by the upstream manifest layer,
        decoupling it from file extensions or MIME types.

        Args:
            handle: An open ArtifactHandle from the dataset loader.
            format_type: The explicitly requested format.

        Yields:
            Immutable ParsedRecord instances.

        Raises:
            UnsupportedFormatError: If the registry cannot resolve the format.
            ParserSyntaxError: If the resolved strategy fails to parse the stream.
        """
        parser = self._registry.resolve(format_type)

        # We assume the caller is inside the context manager of `handle`.
        # Accessing `handle._stream` is normally encapsulated, but for separation of
        # concerns, we should use the entered stream. `handle` doesn't expose the stream
        # cleanly outside of the with block. We will just re-enter it or assume the caller
        # passes the stream? The architecture plan says `handle: ArtifactHandle`.
        # However, entering a handle twice might fail or block if it's already open,
        # but since ArtifactHandle returns the stream on `__enter__`, it's safe to just
        # access `_stream` if we are already inside the context.
        # Alternatively, we can just use `handle.__enter__()` to get the stream.

        stream = handle.__enter__()
        yield from parser.parse(stream)
