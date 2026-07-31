"""Dataset implementations for offline benchmarking."""

from typing import Any

from src.core.benchmark.base import BaseBenchmarkDataset


class LocalBenchmarkDataset(BaseBenchmarkDataset):
    """Generic offline dataset implementation supporting local JSON/dictionary data loading."""

    def __init__(
        self,
        dataset_name: str,
        samples: tuple[dict[str, Any], ...],
        version: str = "1.0",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._dataset_name = dataset_name
        self._samples = samples
        self._version = version
        self._metadata = metadata or {}

    def load_samples(self) -> tuple[dict[str, Any], ...]:
        return self._samples

    def dataset_metadata(self) -> dict[str, Any]:
        return self._metadata

    def dataset_version(self) -> str:
        return self._version


class FEVERDataset(LocalBenchmarkDataset):
    """FEVER dataset implementation for benchmarking."""

    def __init__(
        self,
        samples: tuple[dict[str, Any], ...],
        version: str = "1.0",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        meta = metadata or {"description": "Fact Extraction and VERification dataset"}
        super().__init__("FEVER", samples, version, meta)


class SciFactDataset(LocalBenchmarkDataset):
    """SciFact dataset implementation for benchmarking."""

    def __init__(
        self,
        samples: tuple[dict[str, Any], ...],
        version: str = "1.0",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        meta = metadata or {"description": "Scientific claim verification dataset"}
        super().__init__("SciFact", samples, version, meta)


class ClimateFEVERDataset(LocalBenchmarkDataset):
    """Climate-FEVER dataset implementation for benchmarking."""

    def __init__(
        self,
        samples: tuple[dict[str, Any], ...],
        version: str = "1.0",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        meta = metadata or {"description": "Climate change claim verification dataset"}
        super().__init__("Climate-FEVER", samples, version, meta)
