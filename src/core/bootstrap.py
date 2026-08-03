"""Application bootstrap and initialization routines."""

import logging
import sys
from typing import Any, Sequence

from src.core.cache.cache_models import RetrievalCacheProfileRegistry
from src.core.config import Settings
from src.core.decision.decision_models import DecisionProfile, DecisionProfileRegistry
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
from src.core.paths import ProjectPaths
from src.core.pipeline.orchestrator import ArbiterPipeline
from src.core.reranking.reranking_models import RerankingProfileRegistry
from src.core.retrieval.benchmarking.benchmark_models import BenchmarkProfileRegistry
from src.core.retrieval.optimization.optimization_models import (
    OptimizationProfileRegistry,
)
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
from src.core.verification.base import BaseNLIModel
from src.core.verification.implementations import NLIVerifier
from src.core.verification.verification_models import (
    NLIVerificationDefinition,
    PassageVerificationInput,
    PassageVerificationScore,
    VerificationProfile,
    VerificationProfileRegistry,
    VerificationVerdict,
)

# Alias for type hinting mapping the implementation plan
AppConfig = Settings


class DummyNLIModel:
    """A dummy NLI model implementation to satisfy DI until actual model loading is implemented."""

    def __init__(self) -> None:
        self.label_map = {
            0: VerificationVerdict.SUPPORTED,
            1: VerificationVerdict.INSUFFICIENT,
            2: VerificationVerdict.CONTRADICTED,
        }

    def predict(
        self, batch: tuple[PassageVerificationInput, ...]
    ) -> tuple[PassageVerificationScore, ...]:
        from src.core.verification.verification_models import PassageVerificationScore

        return tuple(
            PassageVerificationScore(
                entailment_probability=1.0,
                contradiction_probability=0.0,
                neutral_probability=0.0,
            )
            for _ in batch
        )


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
        DenseCandidateGenerator,
        DenseRetriever,
        FAISSVectorStore,
        SentenceTransformerQueryEncoder,
    )
    from src.core.retrieval.retrieval_models import DenseRetrievalDefinition

    manifest_path = ProjectPaths.DATA_INDEX / "index_manifest.json"
    if not manifest_path.exists():
        raise RetrievalConfigurationError("Missing index_manifest.json.")

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = IndexManifest.model_validate_json(f.read())
    except Exception as e:
        raise RetrievalConfigurationError(f"Invalid manifest: {e}")

    # For C1.4 we require sparse_index and metadata artifacts
    if (
        "sparse_index" not in manifest.artifacts
        or "metadata" not in manifest.artifacts
        or "dense_index" not in manifest.artifacts
    ):
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
    if (
        not hasattr(manifest, "embedding_metadata")
        or manifest.embedding_metadata is None
    ):
        raise RetrievalConfigurationError("Manifest missing embedding_metadata.")

    # Initialize Dense Components
    # In a real app we'd load device from config, but we'll use "cpu" for fail-fast
    query_encoder = SentenceTransformerQueryEncoder(
        model_id=manifest.embedding_metadata.model_id,
        device="cpu",
    )

    if (
        query_encoder.embedding_dimension
        != manifest.embedding_metadata.embedding_dimension
    ):
        raise RetrievalConfigurationError(
            "Encoder embedding dimension does not match manifest."
        )

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

    from src.core.retrieval.hybrid import HybridRetriever
    from src.core.retrieval.retrieval_models import HybridRetrievalDefinition

    dense_definition = DenseRetrievalDefinition(top_k=5)
    dense_profile = RetrievalProfile(
        profile_id="dense_retrieval",
        definition=dense_definition,
        strategy=dense_engine,
    )

    hybrid_engine = HybridRetriever(
        bm25_generator=generator,
        dense_generator=dense_generator,
        document_store=document_store,
    )

    hybrid_definition = HybridRetrievalDefinition(
        bm25_definition=BM25RetrievalDefinition(top_k=5),
        dense_definition=DenseRetrievalDefinition(top_k=5),
        top_k=5,
        rrf_k=60,
    )
    hybrid_profile = RetrievalProfile(
        profile_id="hybrid_retrieval",
        definition=hybrid_definition,
        strategy=hybrid_engine,
    )

    return RetrievalProfileRegistry(
        profiles=(bm25_profile, dense_profile, hybrid_profile)
    )


def build_verification_registry(config: AppConfig) -> VerificationProfileRegistry:
    """Builds the verification registry with fail-fast validation checks."""
    from src.core.exceptions import VerificationConfigurationError
    from src.core.verification.aggregation import MaxConfidenceAggregationStrategy
    from src.core.verification.implementations import DefaultMetadataProvider
    from src.core.verification.verification_models import (
        ProbabilitySchema,
        VerificationProfileRegistry,
    )

    # 1. Register verifiers
    # Decide if we load DummyNLIModel (fallback for test env) or actual TransformerNLIModel
    engine_model: BaseNLIModel
    if (
        config.environment == "test"
        and config.nli.model_id == "cross-encoder/nli-distilroberta-base"
    ):
        engine_model = DummyNLIModel()
    else:
        from src.core.verification.nli_model import TransformerNLIModel
        from src.core.verification.verification_models import (
            ExecutionDevice,
            NLILabelSchema,
            NLIModelDefinition,
        )

        dev_str = str(config.nli.device).upper()
        if dev_str == "CPU":
            device_enum = ExecutionDevice.CPU
        elif dev_str in ("CUDA", "GPU"):
            device_enum = ExecutionDevice.CUDA
        elif dev_str == "MPS":
            device_enum = ExecutionDevice.MPS
        elif dev_str == "TPU":
            device_enum = ExecutionDevice.TPU
        else:
            device_enum = ExecutionDevice.OTHER

        model_def = NLIModelDefinition(
            model_identifier=config.nli.model_id,
            tokenizer_identifier=config.nli.tokenizer_id,
            execution_device=device_enum,
            inference_precision=config.nli.precision,
            max_sequence_length=config.nli.max_sequence_length,
            batch_size=config.nli.batch_size,
        )

        label_schema = NLILabelSchema(
            label_ordering=("CONTRADICTED", "SUPPORTED", "INSUFFICIENT"),
            id_mapping={0: "CONTRADICTED", 1: "SUPPORTED", 2: "INSUFFICIENT"},
        )

        try:
            engine_model = TransformerNLIModel(
                config=model_def, label_schema=label_schema
            )
            if isinstance(engine_model, TransformerNLIModel):
                if hasattr(engine_model.inference_engine.model, "config"):
                    num_labels = getattr(
                        engine_model.inference_engine.model.config, "num_labels", None
                    )
                    if num_labels is not None and num_labels != 3:
                        raise VerificationConfigurationError(
                            f"NLI model output dimension must be 3, got {num_labels}"
                        )
        except Exception as e:
            raise VerificationConfigurationError(
                f"Failed to load NLI model: {e}"
            ) from e

    engine = NLIVerifier(model=engine_model, strategy_id="default_nli")

    # 2. Register metadata providers
    metadata_provider = DefaultMetadataProvider(model_id="nli-default")

    # 3. Register probability schema
    prob_schema = ProbabilitySchema(
        supported_labels=("SUPPORTED", "CONTRADICTED", "INSUFFICIENT"),
        probability_ordering=("SUPPORTED", "CONTRADICTED", "INSUFFICIENT"),
        tolerance=1e-5,
    )

    # 4. Register aggregation strategies and profiles
    from src.core.verification.aggregation import DefaultEvidenceWeigher
    from src.core.verification.aggregation_strategies import (
        ConsensusAggregationStrategy,
        ContradictionAwareAggregationStrategy,
        WeightedVotingAggregationStrategy,
    )
    from src.core.verification.verification_models import (
        AggregationProfile,
        AggregationProfileRegistry,
        AggregationStrategyType,
    )

    weigher = DefaultEvidenceWeigher()

    consensus_thresh = config.aggregation.consensus_threshold
    contra_thresh = config.aggregation.contradiction_threshold

    if not (0.0 <= consensus_thresh <= 1.0):
        raise VerificationConfigurationError(
            f"Invalid consensus threshold: {consensus_thresh}"
        )
    if not (0.0 <= contra_thresh <= 1.0):
        raise VerificationConfigurationError(
            f"Invalid contradiction threshold: {contra_thresh}"
        )

    max_conf_strat = MaxConfidenceAggregationStrategy(evidence_weigher=weigher)
    weighted_voting_strat = WeightedVotingAggregationStrategy(evidence_weigher=weigher)
    consensus_strat = ConsensusAggregationStrategy(
        evidence_weigher=weigher,
        consensus_threshold=consensus_thresh,
    )
    contra_aware_strat = ContradictionAwareAggregationStrategy(
        evidence_weigher=weigher,
        contradiction_threshold=contra_thresh,
    )

    p_max_conf = AggregationProfile(
        profile_id="max_confidence",
        strategy_type=AggregationStrategyType.MAX_CONFIDENCE,
        strategy=max_conf_strat,
        evidence_weigher=weigher,
    )
    p_weighted_voting = AggregationProfile(
        profile_id="weighted_voting",
        strategy_type=AggregationStrategyType.WEIGHTED_VOTING,
        strategy=weighted_voting_strat,
        evidence_weigher=weigher,
    )
    p_consensus = AggregationProfile(
        profile_id="consensus",
        strategy_type=AggregationStrategyType.CONSENSUS,
        strategy=consensus_strat,
        evidence_weigher=weigher,
    )
    p_contra_aware = AggregationProfile(
        profile_id="contradiction_aware",
        strategy_type=AggregationStrategyType.CONTRADICTION_AWARE,
        strategy=contra_aware_strat,
        evidence_weigher=weigher,
    )

    try:
        agg_registry = AggregationProfileRegistry(
            profiles=(p_max_conf, p_weighted_voting, p_consensus, p_contra_aware)
        )
    except Exception as e:
        raise VerificationConfigurationError(
            f"Aggregation registry validation failed: {e}"
        ) from e

    # 5. Resolve active strategy configuration
    default_strategy_name = config.aggregation.default_strategy
    try:
        agg_profile = agg_registry.resolve(default_strategy_name.lower())
    except KeyError:
        try:
            agg_profile = agg_registry.resolve(default_strategy_name)
        except KeyError:
            agg_profile = p_max_conf

    # 6. Construct definitions
    definition = NLIVerificationDefinition(
        top_k=5,
        probability_schema=prob_schema,
        aggregation_strategy=agg_profile.strategy,
    )

    # Build and validate profile
    try:
        profile = VerificationProfile(
            profile_id="default_verification",
            definition=definition,
            verifier=engine,
            metadata_provider=metadata_provider,
        )
    except Exception as e:
        raise VerificationConfigurationError(
            f"Verification profile initialization failed: {e}"
        ) from e

    # Create and return registry (which validates profile uniqueness internally)
    try:
        registry = VerificationProfileRegistry(profiles=(profile,))
    except Exception as e:
        raise VerificationConfigurationError(
            f"Verification registry initialization failed: {e}"
        ) from e

    return registry


def build_failure_analysis_registry(
    config: AppConfig,
) -> Any:
    """Builds the failure analysis registry."""
    from src.core.exceptions import OptimizationConfigurationError
    from src.core.failure.failure_models import (
        FailureAnalysisDefinition,
        FailureAnalysisProfile,
        FailureAnalysisProfileRegistry,
    )
    from src.core.failure.implementations import (
        CalibrationFailureAnalyzer,
        CompositeFailureAnalyzer,
        DefaultFailureAggregationStrategy,
        InfrastructureFailureAnalyzer,
        RetrievalFailureAnalyzer,
        VerificationFailureAnalyzer,
    )

    # 1. Instantiate specialized analyzers
    r_analyzer = RetrievalFailureAnalyzer()
    v_analyzer = VerificationFailureAnalyzer()
    c_analyzer = CalibrationFailureAnalyzer()
    i_analyzer = InfrastructureFailureAnalyzer()

    analyzers = (r_analyzer, v_analyzer, c_analyzer, i_analyzer)

    # 2. Duplicate analyzer ID checks
    analyzer_ids = [a.runtime_metadata.analyzer_id for a in analyzers]
    if len(analyzer_ids) != len(set(analyzer_ids)):
        raise OptimizationConfigurationError(
            "Duplicate analyzer IDs detected during bootstrap."
        )

    # 3. Instantiate Composite and Aggregation
    agg_strategy = DefaultFailureAggregationStrategy()
    composite = CompositeFailureAnalyzer(
        analyzers=analyzers, aggregation_strategy=agg_strategy
    )

    definition = FailureAnalysisDefinition()

    profile = FailureAnalysisProfile(
        profile_id="default_failure_analysis",
        definition=definition,
        analyzer=composite,
    )

    try:
        registry = FailureAnalysisProfileRegistry(profiles=(profile,))
    except Exception as e:
        raise OptimizationConfigurationError(
            f"Failure analysis registry initialization failed: {e}"
        ) from e

    return registry


def build_failure_correlation_registry(config: AppConfig) -> Any:
    """Builds the failure correlation registry and instantiates data-driven rules."""
    from src.core.exceptions import OptimizationConfigurationError
    from src.core.failure.correlation import DefaultFailureCorrelationStrategy
    from src.core.failure.failure_models import (
        FailureCategory,
        FailureCorrelationDefinition,
        FailureCorrelationProfile,
        FailureCorrelationProfileRegistry,
        FailureCorrelationRule,
    )

    # 1. Instantiate default correlation rules
    rule1 = FailureCorrelationRule(
        rule_id="retrieval_to_verification",
        source_category=FailureCategory.RETRIEVAL,
        target_category=FailureCategory.VERIFICATION,
        precedence=1,
        enabled=True,
    )
    rule2 = FailureCorrelationRule(
        rule_id="verification_to_calibration",
        source_category=FailureCategory.VERIFICATION,
        target_category=FailureCategory.CALIBRATION,
        precedence=1,
        enabled=True,
    )
    rule3 = FailureCorrelationRule(
        rule_id="infrastructure_to_verification",
        source_category=FailureCategory.INFRASTRUCTURE,
        target_category=FailureCategory.VERIFICATION,
        precedence=1,
        enabled=True,
    )

    rules = (rule1, rule2, rule3)

    # Validate duplicate rule IDs
    rule_ids = [r.rule_id for r in rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise OptimizationConfigurationError(
            "Duplicate correlation rule IDs detected during bootstrap."
        )

    # 2. Instantiate strategy
    strategy = DefaultFailureCorrelationStrategy()

    definition = FailureCorrelationDefinition()

    profile = FailureCorrelationProfile(
        profile_id="default_failure_correlation",
        definition=definition,
        rules=rules,
        strategy=strategy,
    )

    try:
        registry = FailureCorrelationProfileRegistry(profiles=(profile,))
    except Exception as e:
        raise OptimizationConfigurationError(
            f"Failure correlation registry initialization failed: {e}"
        ) from e

    return registry


def build_root_cause_registry(config: AppConfig) -> Any:
    """Builds the root cause attribution profile registry for M3.5."""
    from src.core.exceptions import OptimizationConfigurationError
    from src.core.failure.attribution import DependencyGraphRootCauseStrategy
    from src.core.failure.failure_models import (
        RootCauseAttributionDefinition,
        RootCauseProfile,
        RootCauseProfileRegistry,
    )
    from src.core.failure.traversal import FailureGraphTraverser  # noqa: F401

    strategy = DependencyGraphRootCauseStrategy()
    definition = RootCauseAttributionDefinition()

    try:
        strategy.validate_compatibility(definition)
    except Exception as e:
        raise OptimizationConfigurationError(
            f"Root cause strategy compatibility validation failed: {e}"
        ) from e

    profile = RootCauseProfile(
        profile_id="default_root_cause",
        definition=definition,
        strategy=strategy,
    )

    try:
        registry = RootCauseProfileRegistry(profiles=(profile,))
    except Exception as e:
        raise OptimizationConfigurationError(
            f"Root cause registry initialization failed: {e}"
        ) from e

    return registry


def build_severity_policy_registry(config: AppConfig) -> Any:
    """Builds the severity policy registry with data-driven SeverityRules for M3.5."""
    from src.core.exceptions import OptimizationConfigurationError
    from src.core.failure.failure_models import (
        FailureCategory,
        FailureSeverity,
        SeverityPolicyDefinition,
        SeverityPolicyProfile,
        SeverityPolicyRegistry,
        SeverityRule,
    )
    from src.core.failure.severity import ThresholdSeverityPolicy

    rules = (
        SeverityRule(
            rule_id="infrastructure_critical",
            category=FailureCategory.INFRASTRUCTURE,
            minimum_confidence=0.0,
            severity=FailureSeverity.CRITICAL,
            escalation_required=True,
            priority=1,
        ),
        SeverityRule(
            rule_id="retrieval_high",
            category=FailureCategory.RETRIEVAL,
            minimum_confidence=0.0,
            severity=FailureSeverity.HIGH,
            escalation_required=False,
            priority=2,
        ),
        SeverityRule(
            rule_id="verification_high",
            category=FailureCategory.VERIFICATION,
            minimum_confidence=0.0,
            severity=FailureSeverity.HIGH,
            escalation_required=False,
            priority=3,
        ),
        SeverityRule(
            rule_id="calibration_medium",
            category=FailureCategory.CALIBRATION,
            minimum_confidence=0.0,
            severity=FailureSeverity.MEDIUM,
            escalation_required=False,
            priority=4,
        ),
        SeverityRule(
            rule_id="unknown_low",
            category=FailureCategory.UNKNOWN,
            minimum_confidence=0.0,
            severity=FailureSeverity.LOW,
            escalation_required=False,
            priority=5,
        ),
    )

    rule_ids = [r.rule_id for r in rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise OptimizationConfigurationError(
            "Duplicate SeverityRule IDs detected during severity policy bootstrap."
        )

    policy = ThresholdSeverityPolicy()
    definition = SeverityPolicyDefinition(rules=rules)

    try:
        policy.validate_compatibility(definition)
    except Exception as e:
        raise OptimizationConfigurationError(
            f"Severity policy compatibility validation failed: {e}"
        ) from e

    profile = SeverityPolicyProfile(
        profile_id="default_severity_policy",
        definition=definition,
        policy=policy,
    )

    try:
        registry = SeverityPolicyRegistry(profiles=(profile,))
    except Exception as e:
        raise OptimizationConfigurationError(
            f"Severity policy registry initialization failed: {e}"
        ) from e

    return registry


def build_failure_benchmark_registry(config: AppConfig) -> Any:
    """Builds the failure benchmark profile registry for M3.6."""
    from src.core.exceptions import OptimizationConfigurationError
    from src.core.failure.benchmark.benchmark_models import (
        FailureBenchmarkDefinition,
        FailureBenchmarkProfile,
        FailureBenchmarkProfileRegistry,
    )
    from src.core.failure.benchmark.metrics import FailureMetricEngine
    from src.core.failure.benchmark.runner import FailureBenchmarkRunner

    metric_engine = FailureMetricEngine()
    runner = FailureBenchmarkRunner(metric_engine=metric_engine)
    definition = FailureBenchmarkDefinition()

    try:
        runner.validate_compatibility(definition)
    except Exception as e:
        raise OptimizationConfigurationError(
            f"Failure benchmark runner compatibility validation failed: {e}"
        ) from e

    profile = FailureBenchmarkProfile(
        profile_id="default_failure_benchmark",
        definition=definition,
        runner=runner,
    )

    try:
        registry = FailureBenchmarkProfileRegistry(profiles=(profile,))
    except Exception as e:
        raise OptimizationConfigurationError(
            f"Failure benchmark registry initialization failed: {e}"
        ) from e

    return registry


def build_failure_explainability_registry(config: AppConfig) -> Any:
    """Builds the failure explainability profile registry for M3.7."""
    from src.core.exceptions import FailureAnalysisConfigurationError
    from src.core.failure.explainability.explanation_models import (
        FailureExplanationDefinition,
        FailureExplanationProfile,
        FailureExplanationProfileRegistry,
        FailureExplanationTemplate,
    )
    from src.core.failure.explainability.implementations import (
        CompositeFailureExplanationStrategy,
        DecisionTraceExplanationStrategy,
        SummaryExplanationStrategy,
    )
    from src.core.failure.explainability.rendering import FailureReportRenderer

    definition = FailureExplanationDefinition(
        strategy="composite",
        verbosity="standard",
        include_root_cause=True,
        include_correlation=True,
        include_severity=True,
        include_benchmark_references=False,
    )

    _template = FailureExplanationTemplate(
        template_id="default_template",
        verbosity="standard",
    )
    _renderer = FailureReportRenderer()

    summary_strategy = SummaryExplanationStrategy()
    trace_strategy = DecisionTraceExplanationStrategy()
    composite_strategy = CompositeFailureExplanationStrategy(
        strategies=(summary_strategy, trace_strategy)
    )

    try:
        composite_strategy.validate_compatibility(definition)
    except Exception as e:
        raise FailureAnalysisConfigurationError(
            f"Failure explainability strategy compatibility validation failed: {e}"
        ) from e

    profile = FailureExplanationProfile(
        profile_id="default_failure_explainability",
        definition=definition,
        strategy=composite_strategy,
    )

    try:
        registry = FailureExplanationProfileRegistry(profiles=(profile,))
    except Exception as e:
        raise FailureAnalysisConfigurationError(
            f"Failure explainability registry initialization failed: {e}"
        ) from e

    return registry


def build_failure_optimization_registry(config: AppConfig) -> Any:
    """Builds the failure optimization profile registry for M3.8."""
    from src.core.exceptions import OptimizationConfigurationError
    from src.core.failure.optimization.controller import FailureOptimizationController
    from src.core.failure.optimization.health import FailureHealthMonitor
    from src.core.failure.optimization.implementations import (
        BoundedSemaphoreConcurrencyLimiter,
        FailureTelemetryCollector,
    )
    from src.core.failure.optimization.optimization_models import (
        FailureOperationalProfile,
        FailureOptimizationDefinition,
        FailureOptimizationProfile,
        FailureOptimizationProfileRegistry,
    )

    definition = FailureOptimizationDefinition(
        batch_size=16,
        max_concurrent_requests=4,
        timeout_ms=5000.0,
        telemetry_enabled=True,
        profiling_enabled=False,
    )

    limiter = BoundedSemaphoreConcurrencyLimiter(
        max_concurrent_requests=definition.max_concurrent_requests
    )
    collector = FailureTelemetryCollector()
    _health_monitor = FailureHealthMonitor()

    controller = FailureOptimizationController(
        definition=definition,
        concurrency_limiter=limiter,
        telemetry_collector=collector,
    )

    try:
        controller.validate_compatibility(definition)
    except Exception as e:
        raise OptimizationConfigurationError(
            f"Failure optimization controller compatibility validation failed: {e}"
        ) from e

    op_profile = FailureOperationalProfile(
        profile_id="default_failure_operational",
        optimization_definition=definition,
        timeout_policy=definition.timeout_ms,
        concurrency_policy=definition.max_concurrent_requests,
    )

    profile = FailureOptimizationProfile(
        profile_id="default_failure_optimization",
        definition=definition,
        controller=controller,
        operational_profile=op_profile,
    )

    try:
        registry = FailureOptimizationProfileRegistry(profiles=(profile,))
    except Exception as e:
        raise OptimizationConfigurationError(
            f"Failure optimization registry initialization failed: {e}"
        ) from e

    return registry


def build_failure_operational_registry(config: AppConfig) -> Any:
    """Builds the failure operational profile for M3.8."""
    from src.core.failure.optimization.optimization_models import (
        FailureOperationalProfile,
        FailureOptimizationDefinition,
    )

    definition = FailureOptimizationDefinition()
    return FailureOperationalProfile(
        profile_id="default_failure_operational",
        optimization_definition=definition,
        timeout_policy=definition.timeout_ms,
        concurrency_policy=definition.max_concurrent_requests,
    )


def build_calibration_registry(config: AppConfig) -> Any:
    """Builds and validates the calibration profile registry."""
    import math

    from src.core.calibration.calibration_models import (
        CalibrationDefinition,
        CalibrationProfile,
        CalibrationProfileRegistry,
        CalibrationStrategyType,
        IsotonicCalibrationParameters,
        PlattScalingParameters,
        TemperatureScalingParameters,
    )
    from src.core.calibration.implementations import (
        ConfidenceMarginEstimator,
        EntropyEstimator,
        IdentityCalibration,
        IsotonicCalibration,
        NormalizedVarianceEstimator,
        PlattScalingCalibration,
        TemperatureScalingCalibration,
    )
    from src.core.exceptions import CalibrationConfigurationError

    # 1. Register estimators
    estimators = {
        "ENTROPY": EntropyEstimator(),
        "CONFIDENCE_MARGIN": ConfidenceMarginEstimator(),
        "NORMALIZED_VARIANCE": NormalizedVarianceEstimator(),
    }

    # 2. Register calibration strategies
    # Define parameters from configuration settings
    t_val = config.calibration.temperature
    if t_val <= 0.0 or math.isnan(t_val) or math.isinf(t_val):
        raise CalibrationConfigurationError(
            f"Invalid temperature config parameter: {t_val}"
        )

    p_slope = config.calibration.platt_slope
    p_intercept = config.calibration.platt_intercept
    if (
        math.isnan(p_slope)
        or math.isinf(p_slope)
        or math.isnan(p_intercept)
        or math.isinf(p_intercept)
    ):
        raise CalibrationConfigurationError("Platt scaling parameters must be finite.")

    # 3. Construct strategies
    temp_params = TemperatureScalingParameters(temperature=t_val)
    temp_def = CalibrationDefinition(
        strategy=CalibrationStrategyType.TEMPERATURE_SCALING,
        parameters=temp_params,
        uncertainty_method="CONFIDENCE_MARGIN",
    )
    temp_strategy = TemperatureScalingCalibration(
        uncertainty_estimator=estimators["CONFIDENCE_MARGIN"]
    )

    platt_params = PlattScalingParameters(slope=p_slope, intercept=p_intercept)
    platt_def = CalibrationDefinition(
        strategy=CalibrationStrategyType.PLATT_SCALING,
        parameters=platt_params,
        uncertainty_method="CONFIDENCE_MARGIN",
    )
    platt_strategy = PlattScalingCalibration(
        uncertainty_estimator=estimators["CONFIDENCE_MARGIN"]
    )

    isotonic_params = IsotonicCalibrationParameters(
        x_thresholds=(0.0, 0.5, 1.0),
        y_values=(0.0, 0.5, 1.0),
    )
    isotonic_def = CalibrationDefinition(
        strategy=CalibrationStrategyType.ISOTONIC_CALIBRATION,
        parameters=isotonic_params,
        uncertainty_method="CONFIDENCE_MARGIN",
    )
    isotonic_strategy = IsotonicCalibration(
        uncertainty_estimator=estimators["CONFIDENCE_MARGIN"]
    )

    identity_def = CalibrationDefinition(
        strategy=CalibrationStrategyType.IDENTITY,
        parameters=None,
        uncertainty_method="CONFIDENCE_MARGIN",
    )
    identity_strategy = IdentityCalibration(
        uncertainty_estimator=estimators["CONFIDENCE_MARGIN"]
    )

    p_identity = CalibrationProfile(
        profile_id="identity",
        definition=identity_def,
        strategy=identity_strategy,
        uncertainty_estimator=estimators["CONFIDENCE_MARGIN"],
    )
    p_temperature = CalibrationProfile(
        profile_id="temperature_scaling",
        definition=temp_def,
        strategy=temp_strategy,
        uncertainty_estimator=estimators["CONFIDENCE_MARGIN"],
    )
    p_platt = CalibrationProfile(
        profile_id="platt_scaling",
        definition=platt_def,
        strategy=platt_strategy,
        uncertainty_estimator=estimators["CONFIDENCE_MARGIN"],
    )
    p_isotonic = CalibrationProfile(
        profile_id="isotonic_calibration",
        definition=isotonic_def,
        strategy=isotonic_strategy,
        uncertainty_estimator=estimators["CONFIDENCE_MARGIN"],
    )

    try:
        registry = CalibrationProfileRegistry(
            profiles=(p_identity, p_temperature, p_platt, p_isotonic)
        )
    except Exception as e:
        raise CalibrationConfigurationError(
            f"Calibration registry validation failed: {e}"
        ) from e

    return registry


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


def build_decision_registry(config: AppConfig) -> Any:
    """Builds the decision registry for M4.1, M4.2, M4.3, and M4.4."""
    from src.core.decision.decision_models import (
        DecisionDefinition,
        DecisionMetricPolicyRegistry,
        RiskPolicyRegistry,
    )
    from src.core.decision.implementations import (
        DecisionPolicyEngine,
        PolicyDecisionStrategy,
    )
    from src.core.decision.policies import (
        CalibratedMetricPolicy,
        CostBenefitRiskPolicy,
        DecisionMetricResolver,
        EntropyMetricPolicy,
        RawMetricPolicy,
        RawRiskPolicy,
        SeverityThresholdRiskPolicy,
    )
    from src.core.exceptions import DecisionConfigurationError

    definition = DecisionDefinition(
        decision_strategy="policy",
        confidence_policy="calibrated",
        uncertainty_policy="threshold_based",
        failure_policy="severity_aware",
        escalation_policy="default",
    )

    try:
        metric_registry = DecisionMetricPolicyRegistry(
            policies=(
                CalibratedMetricPolicy(),
                RawMetricPolicy(),
                EntropyMetricPolicy(),
            )
        )
        metric_resolver = DecisionMetricResolver(registry=metric_registry)
    except Exception as e:
        raise DecisionConfigurationError(
            f"Decision metric registry initialization failed: {e}"
        ) from e

    try:
        risk_registry = RiskPolicyRegistry(
            policies=(
                RawRiskPolicy(),
                SeverityThresholdRiskPolicy(),
                CostBenefitRiskPolicy(),
            )
        )
    except Exception as e:
        raise DecisionConfigurationError(
            f"Decision risk registry initialization failed: {e}"
        ) from e

    policy_engine = DecisionPolicyEngine(
        metric_resolver=metric_resolver, risk_policy_registry=risk_registry
    )
    policy_groups = PolicyDecisionStrategy.default_policy_groups()
    strategy = PolicyDecisionStrategy(
        policy_groups=policy_groups,
        policy_engine=policy_engine,
    )

    try:
        policy_engine.validate_compatibility(definition)
        strategy.validate_compatibility(definition)
    except Exception as e:
        raise DecisionConfigurationError(
            f"Decision strategy compatibility validation failed: {e}"
        ) from e

    profile = DecisionProfile(
        profile_id="default_decision",
        definition=definition,
        strategy=strategy,
    )

    try:
        registry = DecisionProfileRegistry(profiles=(profile,))
    except Exception as e:
        raise DecisionConfigurationError(
            f"Decision registry initialization failed: {e}"
        ) from e

    return registry


def build_decision_benchmark_registry(config: AppConfig) -> Any:
    """Builds and validates the decision benchmark profile registry for M4.5."""
    from src.core.decision import DecisionContext
    from src.core.decision.benchmark.benchmark_models import (
        DecisionBenchmarkDataset,
        DecisionBenchmarkItem,
        DecisionBenchmarkProfile,
        DecisionBenchmarkProfileRegistry,
        DecisionBenchmarkSuite,
    )
    from src.core.exceptions import DecisionConfigurationError

    # 1. Construct default benchmark dataset
    item1 = DecisionBenchmarkItem(
        item_id="item_default_1",
        context=DecisionContext(),
        expected_action="ABSTAIN",
    )
    dataset = DecisionBenchmarkDataset(
        dataset_id="default_decision_dataset",
        items=(item1,),
    )

    # 2. Construct benchmark suite
    suite = DecisionBenchmarkSuite(
        suite_id="default_decision_suite",
        dataset=dataset,
    )

    # 4. Construct benchmark profile
    profile = DecisionBenchmarkProfile(
        profile_id="default_decision_benchmark",
        enabled_metrics=(
            "accuracy",
            "abstention_rate",
            "escalation_rate",
            "mean_latency_ms",
            "throughput_qps",
        ),
        suite_id=suite.suite_id,
    )

    # 5. Construct registry and validate compatibility
    try:
        registry = DecisionBenchmarkProfileRegistry(profiles=(profile,))
        registry.validate_compatibility(suite.suite_id)
    except Exception as e:
        raise DecisionConfigurationError(
            f"Decision benchmark registry initialization failed: {e}"
        ) from e

    return registry


def build_decision_explanation_registry(config: AppConfig) -> Any:
    """Builds and validates the decision explainability profile registry for M4.6."""
    from src.core.decision.explainability.explainability_models import (
        DecisionExplanationDefinition,
        DecisionExplanationProfile,
        DecisionExplanationProfileRegistry,
    )
    from src.core.decision.explainability.strategies import (
        CompositeExplanationStrategy,
        SummaryExplanationStrategy,
        TraceAuditExplanationStrategy,
    )
    from src.core.exceptions import DecisionConfigurationError

    definition = DecisionExplanationDefinition(
        template_format="markdown",
        include_traces=True,
        include_risk_factors=True,
    )

    summary_strat = SummaryExplanationStrategy()
    trace_strat = TraceAuditExplanationStrategy()
    composite_strat = CompositeExplanationStrategy()

    # Verify compatibility
    try:
        summary_strat.validate_compatibility(definition)
        trace_strat.validate_compatibility(definition)
        composite_strat.validate_compatibility(definition)
    except Exception as e:
        raise DecisionConfigurationError(
            f"Decision explanation strategy compatibility check failed: {e}"
        ) from e

    profile = DecisionExplanationProfile(
        profile_id="default_decision_explanation",
        definition=definition,
        strategy=composite_strat,
    )

    try:
        registry = DecisionExplanationProfileRegistry(profiles=(profile,))
        registry.validate_compatibility(definition)
    except Exception as e:
        raise DecisionConfigurationError(
            f"Decision explanation registry initialization failed: {e}"
        ) from e

    return registry


def build_explanation_registry(
    config: AppConfig,
    verification_registry: Any = None,
    calibration_registry: Any = None,
) -> ExplanationProfileRegistry:
    """Builds and validates the explanation registry."""
    from src.core.exceptions import ExplanationConfigurationError
    from src.core.explainability.explanation_models import (
        VerificationExplanationDefinition,
    )
    from src.core.explainability.implementations import (
        CompositeExplanationStrategy,
        ConfidenceExplanationStrategy,
        DecisionTraceStrategy,
        EvidenceAttributionStrategy,
    )

    engine_m13 = RuleBasedExplainer()
    definition_m13 = RuleBasedExplanationDefinition()
    profile_m13 = ExplanationProfile(
        profile_id="default_explanation",
        definition=definition_m13,
        engine=engine_m13,
    )

    strat_comp = CompositeExplanationStrategy()
    def_comp = VerificationExplanationDefinition(explanation_strategy="COMPOSITE")
    prof_comp = ExplanationProfile(
        profile_id="composite_explanation",
        definition=def_comp,
        engine=strat_comp,
        verification_profile_id="default_verification",
        calibration_profile_id="identity",
    )

    strat_attr = EvidenceAttributionStrategy()
    def_attr = VerificationExplanationDefinition(
        explanation_strategy="EVIDENCE_ATTRIBUTION"
    )
    prof_attr = ExplanationProfile(
        profile_id="evidence_attribution",
        definition=def_attr,
        engine=strat_attr,
        verification_profile_id="default_verification",
        calibration_profile_id="identity",
    )

    strat_trace = DecisionTraceStrategy()
    def_trace = VerificationExplanationDefinition(explanation_strategy="DECISION_TRACE")
    prof_trace = ExplanationProfile(
        profile_id="decision_trace",
        definition=def_trace,
        engine=strat_trace,
        verification_profile_id="default_verification",
        calibration_profile_id="identity",
    )

    strat_conf = ConfidenceExplanationStrategy()
    def_conf = VerificationExplanationDefinition(
        explanation_strategy="CONFIDENCE_EXPLANATION"
    )
    prof_conf = ExplanationProfile(
        profile_id="confidence_explanation",
        definition=def_conf,
        engine=strat_conf,
        verification_profile_id="default_verification",
        calibration_profile_id="identity",
    )

    profiles = (profile_m13, prof_comp, prof_attr, prof_trace, prof_conf)

    if verification_registry is not None:
        for p in profiles:
            if p.verification_profile_id is not None:
                try:
                    verification_registry.resolve(p.verification_profile_id)
                except KeyError as e:
                    raise ExplanationConfigurationError(
                        f"Incompatible verification profile '{p.verification_profile_id}' for explanation profile '{p.profile_id}'"
                    ) from e

    if calibration_registry is not None:
        for p in profiles:
            if p.calibration_profile_id is not None:
                try:
                    calibration_registry.resolve(p.calibration_profile_id)
                except KeyError as e:
                    raise ExplanationConfigurationError(
                        f"Incompatible calibration profile '{p.calibration_profile_id}' for explanation profile '{p.profile_id}'"
                    ) from e

    try:
        registry = ExplanationProfileRegistry(profiles=profiles)
    except Exception as e:
        raise ExplanationConfigurationError(
            f"Explanation registry validation failed: {e}"
        ) from e

    return registry


class DummyCrossEncoderScorer:
    """Dummy CrossEncoderScorer for bootstrap DI testing."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [1.0 for _ in passages]


def build_reranking_registry(config: AppConfig) -> RerankingProfileRegistry:
    """Builds the reranking profile registry."""
    from src.core.reranking.implementations import CrossEncoderReranker
    from src.core.reranking.reranking_models import (
        RerankingDefinition,
        RerankingProfile,
        RerankingProfileRegistry,
    )

    scorer = DummyCrossEncoderScorer()
    engine = CrossEncoderReranker(scorer=scorer)
    definition = RerankingDefinition(top_k_input=10, top_k_output=5)
    profile = RerankingProfile(
        profile_id="default_reranking",
        definition=definition,
        strategy=engine,
    )
    return RerankingProfileRegistry(profiles=(profile,))


def build_cache_registry(config: AppConfig) -> RetrievalCacheProfileRegistry:
    """Builds the retrieval cache registry."""
    from src.core.cache import (
        CacheDefinition,
        InMemoryRetrievalCache,
        RetrievalCacheProfile,
        RetrievalCacheProfileRegistry,
    )

    definition = CacheDefinition(
        enabled=True,
        backend="in_memory",
        ttl_seconds=3600,
        max_entries=1000,
        eviction_policy="lru",
        cache_schema_version="1.0",
    )
    cache_strategy = InMemoryRetrievalCache(definition=definition)
    profile = RetrievalCacheProfile(
        profile_id="default_cache",
        definition=definition,
        strategy=cache_strategy,
    )
    return RetrievalCacheProfileRegistry(profiles=(profile,))


def build_benchmark_registry(config: AppConfig) -> BenchmarkProfileRegistry:
    """Builds the benchmark profile registry."""
    from src.core.retrieval.benchmarking import (
        BenchmarkDefinition,
        BenchmarkProfile,
        BenchmarkProfileRegistry,
        HitRateCalculator,
        MetricRegistry,
        MRRCalculator,
        NDCGCalculator,
        PrecisionCalculator,
        RecallCalculator,
        RetrievalEvaluator,
    )

    metric_registry = MetricRegistry(
        calculators={
            "recall": RecallCalculator(),
            "precision": PrecisionCalculator(),
            "mrr": MRRCalculator(),
            "ndcg": NDCGCalculator(),
            "hit_rate": HitRateCalculator(),
        }
    )
    evaluator = RetrievalEvaluator(metric_registry=metric_registry)
    definition = BenchmarkDefinition(top_k=5)
    profile = BenchmarkProfile(
        profile_id="default_benchmark",
        definition=definition,
        strategy=evaluator,
    )
    return BenchmarkProfileRegistry(profiles=(profile,))


def build_verification_benchmark_registry(config: AppConfig) -> Any:
    """Builds and validates the verification benchmark profile registry."""
    from src.core.benchmark.benchmark_models import (
        BenchmarkDefinition,
        BenchmarkMetricType,
        BenchmarkProfile,
        BenchmarkProfileRegistry,
    )
    from src.core.exceptions import BenchmarkConfigurationError

    selected_metrics = (
        BenchmarkMetricType.ACCURACY,
        BenchmarkMetricType.PRECISION,
        BenchmarkMetricType.RECALL,
        BenchmarkMetricType.F1,
        BenchmarkMetricType.MACRO_F1,
        BenchmarkMetricType.MICRO_F1,
        BenchmarkMetricType.ECE,
        BenchmarkMetricType.MCE,
        BenchmarkMetricType.BRIER_SCORE,
        BenchmarkMetricType.NEGATIVE_LOG_LIKELIHOOD,
        BenchmarkMetricType.MEAN_LATENCY,
        BenchmarkMetricType.P95_LATENCY,
        BenchmarkMetricType.P99_LATENCY,
        BenchmarkMetricType.THROUGHPUT,
        BenchmarkMetricType.ABSTENTION_RATE,
        BenchmarkMetricType.LOW_CONFIDENCE_RATE,
        BenchmarkMetricType.CONFLICT_RATE,
    )

    default_def = BenchmarkDefinition(
        benchmark_name="default_verification_benchmark",
        dataset_identifier="FEVER",
        selected_metrics=selected_metrics,
        evaluation_profile_id="default_verification",
    )

    profile = BenchmarkProfile(
        profile_id="default_benchmark",
        definition=default_def,
    )

    try:
        registry = BenchmarkProfileRegistry(profiles=(profile,))
    except Exception as e:
        raise BenchmarkConfigurationError(
            f"Benchmark registry validation failed: {e}"
        ) from e

    return registry


def build_optimization_registry(config: AppConfig) -> OptimizationProfileRegistry:
    """Builds the optimization profile registry."""
    from src.core.retrieval.optimization import (
        BoundedSemaphoreConcurrencyLimiter,
        ExecutionPolicy,
        OptimizationDefinition,
        OptimizationProfile,
        OptimizationProfileRegistry,
    )

    policy = ExecutionPolicy(
        retrieval_batch_size=16,
        reranking_batch_size=8,
        max_concurrent_requests=4,
        request_timeout_ms=5000.0,
        document_prefetch_size=32,
    )
    limiter = BoundedSemaphoreConcurrencyLimiter(
        max_concurrency=policy.max_concurrent_requests
    )
    definition = OptimizationDefinition(execution_policy=policy)
    profile = OptimizationProfile(
        profile_id="default_optimization",
        definition=definition,
        execution_policy=policy,
        concurrency_limiter=limiter,
    )
    return OptimizationProfileRegistry(profiles=(profile,))


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
    ver_reg = build_verification_registry(config)
    cal_reg = build_calibration_registry(config)
    return ArbiterPipeline(
        retrieval_registry=build_retrieval_registry(config),
        verification_registry=ver_reg,
        failure_analysis_registry=build_failure_analysis_registry(config),
        uncertainty_registry=build_uncertainty_registry(config),
        decision_registry=build_decision_registry(config),
        explanation_registry=build_explanation_registry(config, ver_reg, cal_reg),
        evaluation_registry=build_evaluation_registry(config),
    )


def build_verification_optimization_registry(
    config: AppConfig,
) -> Any:
    """Builds and validates the verification optimization registry."""
    from src.core.exceptions import OptimizationConfigurationError
    from src.core.verification.optimization.optimization_models import (
        VerificationOptimizationDefinition,
        VerificationOptimizationProfile,
        VerificationOptimizationProfileRegistry,
    )

    # 1. Define configurations
    definition = VerificationOptimizationDefinition()

    # 2. Synchronous startup validation
    if definition.request_timeout_ms <= 0:
        raise OptimizationConfigurationError("timeout must be > 0.")
    if definition.verifier_batch_size <= 0:
        raise OptimizationConfigurationError("verifier_batch_size must be > 0.")
    if definition.aggregation_batch_size <= 0:
        raise OptimizationConfigurationError("aggregation_batch_size must be > 0.")
    if definition.calibration_batch_size <= 0:
        raise OptimizationConfigurationError("calibration_batch_size must be > 0.")
    if definition.explanation_batch_size <= 0:
        raise OptimizationConfigurationError("explanation_batch_size must be > 0.")
    if definition.max_concurrent_requests <= 0:
        raise OptimizationConfigurationError("max_concurrent_requests must be > 0.")
    if definition.prefetch_size <= 0:
        raise OptimizationConfigurationError("prefetch_size must be > 0.")
    if not isinstance(definition.telemetry_enabled, bool):
        raise OptimizationConfigurationError("telemetry_enabled must be a boolean.")

    profile = VerificationOptimizationProfile(
        profile_id="default_optimization",
        definition=definition,
    )

    try:
        registry = VerificationOptimizationProfileRegistry(profiles=(profile,))
    except Exception as e:
        raise OptimizationConfigurationError(
            f"Verification optimization registry validation failed: {e}"
        ) from e

    return registry


def build_verification_operational_registry(
    config: AppConfig,
) -> Any:
    """Builds and validates the verification operational registry."""
    from src.core.exceptions import OptimizationConfigurationError
    from src.core.verification.operational.operational_models import (
        VerificationOperationalProfile,
        VerificationOperationalRegistry,
    )

    # 1. Read configuration settings
    env = getattr(config, "environment", "development")
    log_cfg = getattr(config, "logging_configuration", {}) or {}
    readiness_cfg = getattr(config, "readiness_configuration", {}) or {}
    telemetry_cfg = getattr(config, "telemetry_configuration", {}) or {}

    # 2. Perform validation checks
    if env not in ("production", "staging", "development"):
        raise OptimizationConfigurationError(
            f"Invalid environment '{env}'. Must be production, staging, or development."
        )

    # Validate registry dependencies and integrity
    profile = VerificationOperationalProfile(
        profile_id="default_operational",
        environment=env,
        logging_configuration=log_cfg,
        readiness_configuration=readiness_cfg,
        telemetry_configuration=telemetry_cfg,
    )

    try:
        registry = VerificationOperationalRegistry(profiles=(profile,))
    except Exception as e:
        raise OptimizationConfigurationError(
            f"Verification operational registry validation failed: {e}"
        ) from e

    return registry
