"""Unit tests for dataset downloader and transport."""

import hashlib
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.datasets.downloader import DatasetDownloader
from src.core.datasets.metadata import DatasetMetadata, DatasetSchema, DatasetSplit
from src.core.datasets.registry import registry
from src.core.datasets.transport import FileEndpoint, TransportConfig
from src.core.exceptions import IntegrityError, RegistryError


@pytest.fixture
def mock_dataset() -> TransportConfig:
    """Fixture providing a mock dataset transport config."""
    # Register in registry to pass validation
    metadata = DatasetMetadata(
        id="mock_data",
        version="1.0",
        description="Mock",
        domain="mock",
        schema_metadata=DatasetSchema(),
        splits=(DatasetSplit.TRAIN,),
    )
    try:
        registry.register_dataset(metadata)
    except RegistryError:
        pass  # Already registered in another test

    return TransportConfig(
        dataset_id="mock_data",
        version="1.0",
        files=(
            FileEndpoint(
                url="http://example.com/mock.txt",
                destination_filename="mock.txt",
                expected_hash=hashlib.sha256(b"mock_content").hexdigest(),
            ),
        ),
    )


@pytest.fixture
def downloader(mock_dataset: TransportConfig) -> DatasetDownloader:
    dl = DatasetDownloader()
    dl.register_transport(mock_dataset)
    return dl


def test_download_success(downloader: DatasetDownloader, tmp_path: Path) -> None:
    """Test successful download and integrity verification."""
    downloader.raw_dir = tmp_path

    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_response.read.side_effect = [b"mock_", b"content", b""]
    mock_response.getcode.return_value = 200

    with patch("urllib.request.urlopen", return_value=mock_response):
        downloader.download("mock_data", "1.0")

    final_path = tmp_path / "mock_data" / "mock.txt"
    assert final_path.exists()
    assert final_path.read_bytes() == b"mock_content"
    assert not final_path.with_name(final_path.name + ".part").exists()


def test_download_idempotency(downloader: DatasetDownloader, tmp_path: Path) -> None:
    """Test that existing valid files are not redownloaded."""
    downloader.raw_dir = tmp_path
    dataset_dir = tmp_path / "mock_data"
    dataset_dir.mkdir(parents=True)

    final_path = dataset_dir / "mock.txt"
    final_path.write_bytes(b"mock_content")

    with patch("urllib.request.urlopen") as mock_urlopen:
        downloader.download("mock_data", "1.0")
        mock_urlopen.assert_not_called()


def test_download_integrity_failure(
    downloader: DatasetDownloader, tmp_path: Path
) -> None:
    """Test that hash mismatches raise IntegrityError and discard the file."""
    downloader.raw_dir = tmp_path

    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_response.read.side_effect = [b"wrong_content", b""]
    mock_response.getcode.return_value = 200

    with patch("urllib.request.urlopen", return_value=mock_response):
        with pytest.raises(IntegrityError):
            downloader.download("mock_data", "1.0")

    part_path = tmp_path / "mock_data" / "mock.txt.part"
    assert not part_path.exists()


def test_download_retry_on_network_error(
    downloader: DatasetDownloader, tmp_path: Path
) -> None:
    """Test that transient network errors trigger retries."""
    downloader.raw_dir = tmp_path

    # Need to patch time.sleep to avoid waiting during tests
    with patch("time.sleep"):
        with patch(
            "urllib.request.urlopen", side_effect=urllib.error.URLError("Timeout")
        ):
            with pytest.raises(urllib.error.URLError):
                downloader.download("mock_data", "1.0")


def test_download_resume(downloader: DatasetDownloader, tmp_path: Path) -> None:
    """Test that existing partial downloads are resumed."""
    downloader.raw_dir = tmp_path
    dataset_dir = tmp_path / "mock_data"
    dataset_dir.mkdir(parents=True)

    part_path = dataset_dir / "mock.txt.part"
    part_path.write_bytes(b"mock_")

    # The cleanup logic removes stale part files.
    # To test resume, we need to bypass cleanup or create the part file *after* cleanup.
    # We will mock _cleanup_stale_parts.

    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_response.read.side_effect = [b"content", b""]
    mock_response.getcode.return_value = 206  # Partial content

    with patch.object(downloader, "_cleanup_stale_parts"):
        with patch(
            "urllib.request.urlopen", return_value=mock_response
        ) as mock_urlopen:
            downloader.download("mock_data", "1.0")

            # Verify the Range header was sent
            req = mock_urlopen.call_args[0][0]
            assert req.headers.get("Range") == "bytes=5-"

    final_path = dataset_dir / "mock.txt"
    assert final_path.read_bytes() == b"mock_content"
