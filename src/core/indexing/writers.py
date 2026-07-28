import hashlib
import os

from src.core.indexing.models import IndexManifest
from src.core.indexing.pipeline import MetadataWriter


def _hash_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class ManifestWriter(MetadataWriter):
    """Concrete implementation for writing the IndexManifest to disk."""

    def write_manifest(
        self, manifest: IndexManifest, output_dir: str
    ) -> tuple[str, str]:
        output_path = os.path.join(output_dir, "index_manifest.json")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))
        return output_path, _hash_file(output_path)
