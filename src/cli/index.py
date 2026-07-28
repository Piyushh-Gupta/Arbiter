import argparse
import json
import os

from src.core.indexing.builders import (
    DenseIndexBuilder,
    MetadataIndexBuilder,
    SparseIndexBuilder,
)
from src.core.indexing.chunking import RecursiveDocumentChunker
from src.core.indexing.loaders import JSONLCorpusLoader
from src.core.indexing.models import IndexManifest
from src.core.indexing.pipeline import IndexingPipeline
from src.core.indexing.validators import ManifestArtifactValidator
from src.core.indexing.writers import ManifestWriter
from src.core.retrieval.bm25 import WhitespaceTokenizer


def build(corpus_path: str, output_dir: str, encoder_module: str = "dummy") -> None:
    # Dummy import for encoder mapping, in real life we'd use a registry or factory
    if encoder_module == "dummy":
        from tests.unit.test_retrieval_interfaces import DummyDocumentEncoder

        encoder = DummyDocumentEncoder()
    else:
        raise NotImplementedError(f"Encoder {encoder_module} not implemented in CLI")

    loader = JSONLCorpusLoader()
    chunker = RecursiveDocumentChunker()
    tokenizer = WhitespaceTokenizer()
    builders = [
        SparseIndexBuilder(tokenizer),
        DenseIndexBuilder(),
        MetadataIndexBuilder(),
    ]
    writer = ManifestWriter()

    # We do a preliminary pass to get the dataset version if we wanted strict validation
    # For build, we just pass expected dimensions based on the loaded encoder
    # The actual dataset version comes from the loader
    validator = ManifestArtifactValidator(
        expected_dataset_version="",  # We'll patch this post-load for the validator
        expected_dimension=encoder.embedding_dimension,
    )

    pipeline = IndexingPipeline(
        loader=loader,
        chunker=chunker,
        encoder=encoder,
        builders=builders,
        writer=writer,
        validator=validator,
    )

    dataset_version, corpus = pipeline._loader.load(corpus_path)
    # Patch the validator for this run
    if isinstance(pipeline._validator, ManifestArtifactValidator):
        pipeline._validator.expected_dataset_version = dataset_version

    chunks = pipeline._chunker.chunk(corpus, dataset_version)
    texts = [c.text for c in chunks]
    embeddings = pipeline._encoder.encode_batch(texts) if texts else None

    from src.core.indexing.models import ArtifactLocation, IndexManifest

    artifacts = {}
    for builder in pipeline._builders:
        path, checksum = builder.build(chunks, embeddings, output_dir)
        artifacts[builder.builder_id] = ArtifactLocation(path=path, checksum=checksum)

    manifest = IndexManifest(
        dataset_version=dataset_version,
        encoder_model_id=pipeline._encoder.model_id,
        embedding_dimension=pipeline._encoder.embedding_dimension,
        artifacts=artifacts,
    )
    pipeline._writer.write_manifest(manifest, output_dir)
    pipeline._validator.validate(manifest)

    print(f"Index built successfully at {output_dir}")
    print(f"Dataset Version: {manifest.dataset_version}")


def validate(manifest_path: str, dataset_version: str, dimension: int) -> None:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = IndexManifest.model_validate_json(f.read())

    validator = ManifestArtifactValidator(
        expected_dataset_version=dataset_version,
        expected_dimension=dimension,
    )
    validator.validate(manifest)
    print("Manifest and artifacts validated successfully.")


def inspect(manifest_path: str) -> None:
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(json.dumps(data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Arbiter Offline Indexing CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a new index")
    build_parser.add_argument("--corpus", required=True, help="Path to JSONL corpus")
    build_parser.add_argument("--output", required=True, help="Output directory")

    validate_parser = subparsers.add_parser(
        "validate", help="Validate an existing index"
    )
    validate_parser.add_argument(
        "--manifest", required=True, help="Path to index_manifest.json"
    )
    validate_parser.add_argument(
        "--version", required=True, help="Expected dataset version (checksum)"
    )
    validate_parser.add_argument(
        "--dim", required=True, type=int, help="Expected embedding dimension"
    )

    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect an existing index manifest"
    )
    inspect_parser.add_argument(
        "--manifest", required=True, help="Path to index_manifest.json"
    )

    args = parser.parse_args()

    if args.command == "build":
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        build(args.corpus, args.output)
    elif args.command == "validate":
        validate(args.manifest, args.version, args.dim)
    elif args.command == "inspect":
        inspect(args.manifest)


if __name__ == "__main__":
    main()
