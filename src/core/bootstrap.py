"""Application bootstrap and initialization routines."""

import logging
import sys
from typing import Sequence

from src.core.config import Settings
from src.core.decision.decision_models import (
    DecisionProfile,
    DecisionProfileRegistry,
    ThresholdDecisionDefinition,
)
from src.core.decision.implementations import ThresholdDecisionEngine
from src.core.evaluation.evaluation_models import (
    EvaluationProfile,
    EvaluationProfileRegistry,
    RuleBasedEvaluationDefinition,
)
from src.core.evaluation.implementations import RuleBasedEvaluator
from src.core.explainability.explainability_models import (
    ExplanationProfile,
    ExplanationProfileRegistry,
    RuleBasedExplanationDefinition,
)
from src.core.explainability.implementations import RuleBasedExplainer
from src.core.failure_analysis.failure_analysis_models import (
    FailureAnalysisProfile,
    FailureAnalysisProfileRegistry,
    VerificationFailureAnalysisDefinition,
)
from src.core.failure_analysis.implementations import VerificationFailureAnalyzer
from src.core.paths import ProjectPaths
from src.core.pipeline.orchestrator import ArbiterPipeline
from src.core.retrieval.retrieval_models import (
    BM25RetrievalDefinition,
    RetrievalProfile,
    RetrievalProfileRegistry,
)
from src.core.uncertainty.implementations import FailureAwareUncertaintyEstimator
from src.core.uncertainty.uncertainty_models import (
    FailureAwareUncertaintyDefinition,
    UncertaintyProfile,
    UncertaintyProfileRegistry,
)
from src.core.validation import validate_startup
from src.core.verification.implementations import NLIVerifier
from src.core.verification.verification_models import (
    NLIVerificationDefinition,
    VerificationLabel,
    VerificationProfile,
    VerificationProfileRegistry,
)

# Alias for type hinting mapping the implementation plan
AppConfig = Settings


class DummyNLIModel:
    """A dummy NLI model implementation to satisfy DI until actual model loading is implemented."""

    def __init__(self) -> None:
        self.label_map = {
            0: VerificationLabel.SUPPORTS,
            1: VerificationLabel.NOT_ENOUGH_INFO,
            2: VerificationLabel.REFUTES,
        }

    def predict(
        self, claim: str, passages: Sequence[str]
    ) -> list[tuple[float, float, float]]:
        return [(1.0, 0.0, 0.0) for _ in passages]


def _create_required_directories() -> None:
    """Explicitly create all required directories defined in paths."""
    for directory in ProjectPaths.get_all_required_dirs():
        directory.mkdir(parents=True, exist_ok=True)


def _configure_logging(config: AppConfig) -> None:
    """Configures structured application logging."""
    # Enforce basic configuration with infrastructure focus
    log_format = (
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        if config.environment == "development"
        else '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
    )
    logging.basicConfig(
        level=config.log.level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def initialize_application(config: AppConfig) -> None:
    """
    Execute the startup orchestration routine.

    This function should be called at the very beginning of the application lifecycle.
    It coordinates directory creation, logging setup, and system validation.
    """
    _configure_logging(config)
    _create_required_directories()
    validate_startup()

    logger = logging.getLogger("arbiter.bootstrap")
    logger.info("Application bootstrap completed successfully.")


def build_retrieval_registry(config: AppConfig) -> RetrievalProfileRegistry:
    """Builds the retrieval registry by loading offline-generated BM25 artifacts."""
    import os
    import pickle

    from src.core.exceptions import RetrievalConfigurationError
    from src.core.indexing.models import IndexManifest
    from src.core.retrieval.bm25 import (
        BM25CandidateGenerator,
        BM25Retriever,
        MetadataDocumentStore,
        WhitespaceTokenizer,
    )
    from src.core.retrieval.dense import (
        SentenceTransformerQueryEncoder,
        FAISSVectorStore,
        DenseCandidateGenerator,
        DenseRetriever,
    )
    from src.core.retrieval.retrieval_models import (
        DenseRetrievalDefinition,
    )

    manifest_path = ProjectPaths.DATA_INDEX / "index_manifest.json"
    if not manifest_path.exists():
        raise RetrievalConfigurationError("Missing index_manifest.json.")

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = IndexManifest.model_validate_json(f.read())
    except Exception as e:
        raise RetrievalConfigurationError(f"Invalid manifest: {e}")

    # For C1.4 we require sparse_index and metadata artifacts
    if "sparse_index" not in manifest.artifacts or "metadata" not in manifest.artifacts or "dense_index" not in manifest.artifacts:
        raise RetrievalConfigurationError(
            "Manifest missing required sparse_index, dense_index or metadata artifacts."
        )

    sparse_path = manifest.artifacts["sparse_index"].path
    metadata_path = manifest.artifacts["metadata"].path

    if not os.path.exists(sparse_path):
        raise RetrievalConfigurationError(f"Missing BM25 artifact at {sparse_path}")
    if not os.path.exists(metadata_path):
        raise RetrievalConfigurationError(
            f"Missing metadata artifact at {metadata_path}"
        )

    # Load BM25Okapi
    try:
        with open(sparse_path, "rb") as f:
            index = pickle.load(f)
    except Exception as e:
        raise RetrievalConfigurationError(f"Failed to load BM25 index: {e}")

    # Load MetadataDocumentStore and extract ordered span_ids
    # We must extract them sequentially to match the BM25 index order
    document_store = MetadataDocumentStore(metadata_path)
    ordered_span_ids = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        import json

        for line in f:
            if line.strip():
                chunk_data = json.loads(line)
                ordered_span_ids.append(chunk_data["span_id"])

    tokenizer = WhitespaceTokenizer()

    # We could theoretically checksum validate here again, but ManifestArtifactValidator
    # runs during validate_startup(). We rely on that for checksum validation.

    generator = BM25CandidateGenerator(
        index=index,
        span_ids=ordered_span_ids,
        tokenizer=tokenizer,
    )

    engine = BM25Retriever(generator=generator, document_store=document_store)

    bm25_definition = BM25RetrievalDefinition(top_k=5)
    bm25_profile = RetrievalProfile(
        profile_id="bm25_retrieval",
        definition=bm25_definition,
        strategy=engine,
    )

    dense_path = manifest.artifacts["dense_index"].path
    if not os.path.exists(dense_path):
        raise RetrievalConfigurationError(f"Missing FAISS artifact at {dense_path}")

    # Validate embedding metadata
    if not hasattr(manifest, "embedding_metadata") or manifest.embedding_metadata is None:
        raise RetrievalConfigurationError("Manifest missing embedding_metadata.")

    # Initialize Dense Components
    # In a real app we'd load device from config, but we'll use "cpu" for fail-fast
    query_encoder = SentenceTransformerQueryEncoder(
        model_id=manifest.embedding_metadata.model_id,
        device="cpu",
    )
    
    if query_encoder.embedding_dimension != manifest.embedding_metadata.embedding_dimension:
        raise RetrievalConfigurationError("Encoder embedding dimension does not match manifest.")

    faiss_store = FAISSVectorStore(
        index_path=dense_path,
        manifest=manifest,
        span_ids=ordered_span_ids,
    )

    dense_generator = DenseCandidateGenerator(
        query_encoder=query_encoder,
        vector_store=faiss_store,
    )

    dense_engine = DenseRetriever(
        candidate_generator=dense_generator,
        document_store=document_store,
    )

    dense_definition = DenseRetrievalDefinition(top_k=5)
    dense_profile = RetrievalProfile(
        profile_id="dense_retrieval",
        definition=dense_definition,
        strategy=dense_engine,
    )

    return RetrievalProfileRegistry(profiles=(bm25_profile, dense_profile))


def build_verification_registry(config: AppConfig) -> VerificationProfileRegistry:
    """Builds the verification registry."""
    engine = NLIVerifier(model=DummyNLIModel(), strategy_id="dummy_nli")
    definition = NLIVerificationDefinition(top_k=5)
    profile = VerificationProfile(
        profile_id="default_verification",
        definition=definition,
        verifier=engine,
    )
    return VerificationProfileRegistry(profiles=(profile,))


def build_failure_analysis_registry(
    config: AppConfig,
) -> FailureAnalysisProfileRegistry:
    """Builds the failure analysis registry."""
    engine = VerificationFailureAnalyzer()
    definition = VerificationFailureAnalysisDefinition(min_confidence_threshold=0.5)
    profile = FailureAnalysisProfile(
        profile_id="default_failure_analysis",
        definition=definition,
        analyzer=engine,
    )
    return FailureAnalysisProfileRegistry(profiles=(profile,))


def build_uncertainty_registry(config: AppConfig) -> UncertaintyProfileRegistry:
    """Builds the uncertainty registry."""
    engine = FailureAwareUncertaintyEstimator()
    definition = FailureAwareUncertaintyDefinition(
        none_threshold=0.1, low_threshold=0.3, medium_threshold=0.6, high_threshold=0.9
    )
    profile = UncertaintyProfile(
        profile_id="default_uncertainty",
        definition=definition,
        estimator=engine,
    )
    return UncertaintyProfileRegistry(profiles=(profile,))


def build_decision_registry(config: AppConfig) -> DecisionProfileRegistry:
    """Builds the decision registry."""
    engine = ThresholdDecisionEngine()
    definition = ThresholdDecisionDefinition(
        accept_max_uncertainty=0.3, reject_max_uncertainty=0.7
    )
    profile = DecisionProfile(
        profile_id="default_decision",
        definition=definition,
        engine=engine,
    )
    return DecisionProfileRegistry(profiles=(profile,))


def build_explanation_registry(config: AppConfig) -> ExplanationProfileRegistry:
    """Builds the explanation registry."""
    engine = RuleBasedExplainer()
    definition = RuleBasedExplanationDefinition()
    profile = ExplanationProfile(
        profile_id="default_explanation",
        definition=definition,
        engine=engine,
    )
    return ExplanationProfileRegistry(profiles=(profile,))


def build_evaluation_registry(config: AppConfig) -> EvaluationProfileRegistry:
    """Builds the evaluation registry."""
    engine = RuleBasedEvaluator()
    definition = RuleBasedEvaluationDefinition()
    profile = EvaluationProfile(
        profile_id="default_evaluation",
        definition=definition,
        engine=engine,
    )
    return EvaluationProfileRegistry(profiles=(profile,))


def build_pipeline(config: AppConfig) -> ArbiterPipeline:
    """Builds the full Arbiter Pipeline."""
    return ArbiterPipeline(
        retrieval_registry=build_retrieval_registry(config),
        verification_registry=build_verification_registry(config),
        failure_analysis_registry=build_failure_analysis_registry(config),
        uncertainty_registry=build_uncertainty_registry(config),
        decision_registry=build_decision_registry(config),
        explanation_registry=build_explanation_registry(config),
        evaluation_registry=build_evaluation_registry(config),
    )
