"""Dataset downloader orchestration."""

import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path

import structlog

from src.core.config import settings
from src.core.datasets.registry import registry
from src.core.datasets.transport import FileEndpoint, TransportConfig
from src.core.exceptions import DownloadError, IntegrityError, RegistryError
from src.core.paths import ProjectPaths
from src.core.utils.retry import with_retry

logger = structlog.get_logger(__name__)


class DatasetDownloader:
    """Orchestrates dataset downloading with idempotency and integrity verification."""

    def __init__(self) -> None:
        self.raw_dir = ProjectPaths.DATA_RAW
        # A simple in-memory mapping for transport configs (to be expanded later)
        self._transports: dict[str, TransportConfig] = {}

    def _get_key(self, dataset_id: str, version: str) -> str:
        return f"{dataset_id}@{version}"

    def register_transport(self, config: TransportConfig) -> None:
        """Register a transport configuration for a dataset."""
        key = self._get_key(config.dataset_id, config.version)
        self._transports[key] = config

    def _resolve_transport(self, dataset_id: str, version: str) -> TransportConfig:
        key = self._get_key(dataset_id, version)
        if key not in self._transports:
            raise DownloadError(f"No transport configuration found for {key}")
        return self._transports[key]

    def _verify_integrity(
        self, file_path: Path, expected_hash: str, algorithm: str
    ) -> bool:
        """Verify the cryptographic integrity of a file."""
        if algorithm.lower() != "sha256":
            raise DownloadError(f"Unsupported hash algorithm: {algorithm}")

        hasher = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
        except OSError as e:
            logger.error(
                "Failed to read file for hashing", path=str(file_path), error=str(e)
            )
            return False

        return hasher.hexdigest() == expected_hash

    def _cleanup_stale_parts(self, target_dir: Path) -> None:
        """Remove orphaned .part files in the target directory."""
        if not target_dir.exists():
            return
        for part_file in target_dir.glob("*.part"):
            try:
                part_file.unlink()
                logger.info("Cleaned up stale part file", path=str(part_file))
            except OSError as e:
                logger.warning(
                    "Failed to clean up part file", path=str(part_file), error=str(e)
                )

    @with_retry(exceptions=(urllib.error.URLError, OSError))
    def _download_file(
        self, endpoint: FileEndpoint, dest_path: Path, auth_headers: dict[str, str]
    ) -> None:
        """Download a single file with resume capability and retry policy."""
        part_path = dest_path.with_name(dest_path.name + ".part")

        headers = dict(auth_headers)

        # Check for existing part file for resume
        initial_size = 0
        if part_path.exists():
            initial_size = part_path.stat().st_size
            if initial_size > 0:
                headers["Range"] = f"bytes={initial_size}-"
                logger.info(
                    "Resuming download", path=str(part_path), initial_size=initial_size
                )

        req = urllib.request.Request(endpoint.url, headers=headers)

        try:
            with urllib.request.urlopen(
                req, timeout=settings.download.timeout_seconds
            ) as response:
                # If server doesn't support range, it returns 200 instead of 206
                if initial_size > 0 and response.getcode() != 206:
                    logger.warning(
                        "Server does not support Range requests, restarting download",
                        url=endpoint.url,
                    )
                    initial_size = 0
                    mode = "wb"
                else:
                    mode = "ab" if initial_size > 0 else "wb"

                with open(part_path, mode) as out_file:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        out_file.write(chunk)
        except urllib.error.HTTPError as e:
            if (
                e.code == 416
            ):  # Range Not Satisfiable (e.g., already fully downloaded but not moved)
                logger.warning(
                    "Range not satisfiable, assuming complete or corrupted, restarting",
                    url=endpoint.url,
                )
                if part_path.exists():
                    part_path.unlink()
                # We raise OSError to trigger retry which will start fresh
                raise OSError("Range not satisfiable") from e
            raise

    def download(self, dataset_id: str, version: str | None = None) -> None:
        """
        Download a dataset into the raw data directory.

        This is an explicit operation that fetches artifacts atomically.
        """
        if version is None:
            version = "1.0.0"  # Fallback to default if not provided

        # Ensure dataset is actually registered in the Dataset Registry (Validation)
        try:
            registry.get_dataset(dataset_id, version)
        except RegistryError as e:
            raise DownloadError(
                f"Cannot download unregistered dataset {dataset_id}@{version}"
            ) from e

        transport = self._resolve_transport(dataset_id, version)
        dataset_dir = self.raw_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        self._cleanup_stale_parts(dataset_dir)

        auth_headers = transport.auth.headers if transport.auth else {}

        for endpoint in transport.files:
            dest_path = dataset_dir / endpoint.destination_filename

            # 1. Idempotency Check
            if dest_path.exists():
                logger.info("Verifying existing file", path=str(dest_path))
                if self._verify_integrity(
                    dest_path, endpoint.expected_hash, endpoint.hash_algorithm
                ):
                    logger.info(
                        "File already exists and integrity verified, skipping",
                        path=str(dest_path),
                    )
                    continue
                else:
                    logger.warning(
                        "Existing file failed integrity check, redownloading",
                        path=str(dest_path),
                    )
                    dest_path.unlink()

            # 2. Download
            logger.info("Starting download", url=endpoint.url, dest=str(dest_path))
            self._download_file(endpoint, dest_path, auth_headers)

            part_path = dest_path.with_name(dest_path.name + ".part")
            if not part_path.exists():
                raise DownloadError(f"Download failed: part file not found {part_path}")

            # 3. Integrity Verification
            logger.info("Verifying downloaded file", path=str(part_path))
            if not self._verify_integrity(
                part_path, endpoint.expected_hash, endpoint.hash_algorithm
            ):
                part_path.unlink()
                raise IntegrityError(
                    f"Integrity verification failed for {endpoint.destination_filename}"
                )

            # 4. Atomic Commit
            os.replace(part_path, dest_path)
            logger.info("Download complete", path=str(dest_path))
