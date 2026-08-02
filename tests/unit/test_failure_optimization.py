"""Unit tests for M3.8 Failure Production Optimization & Hardening subsystem."""

import concurrent.futures

import pytest
from pydantic import ValidationError

from src.core.bootstrap import (
    build_failure_operational_registry,
    build_failure_optimization_registry,
)
from src.core.config import Settings
from src.core.exceptions import (
    DuplicateFailureAnalysisProfileError,
    FailureAnalysisProfileNotFoundError,
    OptimizationTimeoutError,
)
from src.core.failure.failure_models import (
    FailureAnalysisInput,
    FailureAnalysisResult,
    FailureCategory,
    FailureClassification,
    FailureDiagnostic,
    FailureRootCause,
    FailureSeverity,
    FailureTrace,
)
from src.core.failure.optimization import (
    BoundedSemaphoreConcurrencyLimiter,
    FailureExecutionMetrics,
    FailureHealthMonitor,
    FailureOperationalProfile,
    FailureOptimizationController,
    FailureOptimizationDefinition,
    FailureOptimizationProfile,
    FailureOptimizationProfileRegistry,
    FailureTelemetryCollector,
    FailureTelemetryRecord,
    FailureTelemetrySnapshot,
)
from src.core.failure_analysis.failure_analysis_models import FailureMetadata
from src.core.failure_analysis.failure_analysis_models import (
    FailureSeverity as LegacyFailureSeverity,
)
from src.core.retrieval.retrieval_models import (
    EvidenceBundle,
    EvidencePassage,
    RetrievalMetadata,
)
from src.core.verification.verification_models import (
    VerificationResult,
    VerificationVerdict,
)


@pytest.fixture
def dummy_evidence_bundle() -> EvidenceBundle:
    p1 = EvidencePassage(document_id="d1", span_id="s1", text="text", score=0.9)
    return EvidenceBundle(
        claim="Test claim",
        passages=(p1,),
        metadata=RetrievalMetadata(strategy_id="test", top_k=1),
    )


@pytest.fixture
def dummy_verification_result(
    dummy_evidence_bundle: EvidenceBundle,
) -> VerificationResult:
    return VerificationResult(
        verdict=VerificationVerdict.SUPPORTED,
        confidence=0.9,
        evidence_bundle=dummy_evidence_bundle,
    )


@pytest.fixture
def dummy_analysis_input(
    dummy_verification_result: VerificationResult,
) -> FailureAnalysisInput:
    from src.core.failure.failure_models import FailureAnalysisDefinition

    return FailureAnalysisInput(
        claim="Test claim",
        pipeline_artifacts={"verification": dummy_verification_result},
        definition=FailureAnalysisDefinition(),
    )


@pytest.fixture
def dummy_analysis_result(
    dummy_verification_result: VerificationResult,
) -> FailureAnalysisResult:
    return FailureAnalysisResult(
        classification=FailureClassification(
            category=FailureCategory.RETRIEVAL,
            severity=FailureSeverity.HIGH,
            affected_subsystem="retrieval",
        ),
        diagnostic=FailureDiagnostic(
            root_cause=FailureRootCause.LOW_RETRIEVAL_RECALL,
            diagnostic_summary="Low retrieval recall",
            affected_artifacts=("index",),
        ),
        trace=FailureTrace(
            analyzer_execution_order=("RetrievalFailureAnalyzer",),
            diagnostic_sequence=("check1",),
            inspected_artifacts=("index",),
        ),
        failure_flags=frozenset(),
        severity=LegacyFailureSeverity.HIGH,
        verification_result=dummy_verification_result,
        metadata=FailureMetadata(strategy_id="test"),
    )


class DummyAnalyzer:
    def __init__(self, result: FailureAnalysisResult) -> None:
        self.result = result

    def analyze(self, input_data: FailureAnalysisInput) -> FailureAnalysisResult:
        return self.result


# --- Model Tests ---


def test_optimization_models_immutability() -> None:
    definition = FailureOptimizationDefinition()
    with pytest.raises(ValidationError):
        setattr(definition, "batch_size", 32)

    metrics = FailureExecutionMetrics(total_latency_ms=15.0)
    with pytest.raises(ValidationError):
        setattr(metrics, "total_latency_ms", 20.0)

    record = FailureTelemetryRecord(
        request_id="req1",
        timestamp="2026-08-01T00:00:00Z",
        execution_metrics=metrics,
    )
    with pytest.raises(ValidationError):
        setattr(record, "success", False)

    snapshot = FailureTelemetrySnapshot()
    with pytest.raises(ValidationError):
        setattr(snapshot, "total_requests", 10)

    op_profile = FailureOperationalProfile(
        profile_id="op1",
        optimization_definition=definition,
    )
    with pytest.raises(ValidationError):
        setattr(op_profile, "timeout_policy", 1000.0)


# --- Concurrency Tests ---


def test_bounded_semaphore_concurrency_limiter() -> None:
    limiter = BoundedSemaphoreConcurrencyLimiter(max_concurrent_requests=2)
    assert limiter.max_capacity == 2
    assert limiter.active_slots == 0

    assert limiter.acquire(timeout_ms=100.0) is True
    assert limiter.active_slots == 1

    assert limiter.acquire(timeout_ms=100.0) is True
    assert limiter.active_slots == 2

    # Third acquisition should timeout
    assert limiter.acquire(timeout_ms=50.0) is False
    assert limiter.active_slots == 2

    limiter.release()
    assert limiter.active_slots == 1

    # Now can acquire again
    assert limiter.acquire(timeout_ms=100.0) is True
    limiter.release()
    limiter.release()
    assert limiter.active_slots == 0


def test_concurrency_limiter_invalid_capacity() -> None:
    with pytest.raises(ValueError):
        BoundedSemaphoreConcurrencyLimiter(max_concurrent_requests=0)


# --- Telemetry Tests ---


def test_failure_telemetry_collector() -> None:
    collector = FailureTelemetryCollector()
    metrics1 = FailureExecutionMetrics(total_latency_ms=10.0)
    metrics2 = FailureExecutionMetrics(total_latency_ms=20.0)

    r1 = FailureTelemetryRecord(
        request_id="r1",
        timestamp="ts1",
        execution_metrics=metrics1,
        success=True,
    )
    r2 = FailureTelemetryRecord(
        request_id="r2",
        timestamp="ts2",
        execution_metrics=metrics2,
        success=False,
    )

    collector.record(r1)
    collector.record(r2)

    records = collector.get_records()
    assert len(records) == 2

    snap = collector.snapshot()
    assert snap.total_requests == 2
    assert snap.failure_count == 1
    assert snap.average_latency_ms == 15.0


def test_telemetry_collector_thread_safety() -> None:
    collector = FailureTelemetryCollector()

    def _worker(idx: int) -> None:
        rec = FailureTelemetryRecord(
            request_id=f"r_{idx}",
            timestamp="ts",
            execution_metrics=FailureExecutionMetrics(total_latency_ms=float(idx)),
        )
        collector.record(rec)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_worker, i) for i in range(50)]
        concurrent.futures.wait(futures)

    assert len(collector.get_records()) == 50
    snap = collector.snapshot()
    assert snap.total_requests == 50


# --- Health Monitor Tests ---


def test_failure_health_monitor() -> None:
    monitor = FailureHealthMonitor()

    def1 = FailureOptimizationDefinition()
    limiter = BoundedSemaphoreConcurrencyLimiter()
    ctrl = FailureOptimizationController(def1, limiter)
    p1 = FailureOptimizationProfile(profile_id="p1", definition=def1, controller=ctrl)
    registry = FailureOptimizationProfileRegistry(profiles=(p1,))

    assert monitor.validate_registry(registry) is True
    assert monitor.validate_registry(None) is False

    readiness = monitor.check_readiness(
        optimization_registry=registry,
        analyzer_registry="dummy_analyzer_registry",
    )
    assert readiness["status"] == "READY"
    assert readiness["registry_valid"] is True
    assert readiness["dependencies_valid"] is True

    health = monitor.get_health_status()
    assert health["status"] == "HEALTHY"


# --- Controller Tests ---


def test_failure_optimization_controller_execution(
    dummy_analysis_input: FailureAnalysisInput,
    dummy_analysis_result: FailureAnalysisResult,
) -> None:
    definition = FailureOptimizationDefinition(timeout_ms=1000.0)
    limiter = BoundedSemaphoreConcurrencyLimiter(max_concurrent_requests=2)
    collector = FailureTelemetryCollector()
    controller = FailureOptimizationController(
        definition=definition,
        concurrency_limiter=limiter,
        telemetry_collector=collector,
    )

    analyzer = DummyAnalyzer(dummy_analysis_result)

    result, metrics = controller.execute(
        input_data=dummy_analysis_input,
        analyzer=analyzer,
    )

    assert result == dummy_analysis_result
    assert metrics.total_latency_ms >= 0.0
    assert len(collector.get_records()) == 1


def test_controller_timeout_enforcement(
    dummy_analysis_input: FailureAnalysisInput,
    dummy_analysis_result: FailureAnalysisResult,
) -> None:
    definition = FailureOptimizationDefinition(timeout_ms=50.0)
    limiter = BoundedSemaphoreConcurrencyLimiter(max_concurrent_requests=1)
    controller = FailureOptimizationController(
        definition=definition,
        concurrency_limiter=limiter,
    )

    # Exhaust the single slot
    assert limiter.acquire(timeout_ms=100.0) is True

    analyzer = DummyAnalyzer(dummy_analysis_result)
    with pytest.raises(OptimizationTimeoutError):
        controller.execute(input_data=dummy_analysis_input, analyzer=analyzer)

    limiter.release()


# --- Registry Tests ---


def test_optimization_registry_duplicate_raises() -> None:
    def1 = FailureOptimizationDefinition()
    limiter = BoundedSemaphoreConcurrencyLimiter()
    ctrl = FailureOptimizationController(def1, limiter)
    p1 = FailureOptimizationProfile(profile_id="p1", definition=def1, controller=ctrl)
    p2 = FailureOptimizationProfile(profile_id="p1", definition=def1, controller=ctrl)

    with pytest.raises(DuplicateFailureAnalysisProfileError):
        FailureOptimizationProfileRegistry(profiles=(p1, p2))


def test_optimization_registry_resolution() -> None:
    def1 = FailureOptimizationDefinition()
    limiter = BoundedSemaphoreConcurrencyLimiter()
    ctrl = FailureOptimizationController(def1, limiter)
    p1 = FailureOptimizationProfile(profile_id="p1", definition=def1, controller=ctrl)
    registry = FailureOptimizationProfileRegistry(profiles=(p1,))

    assert registry.resolve("p1").profile_id == "p1"

    with pytest.raises(FailureAnalysisProfileNotFoundError):
        registry.resolve("missing_profile")


# --- Bootstrap Tests ---


def test_bootstrap_failure_optimization_registries() -> None:
    config = Settings()
    opt_registry = build_failure_optimization_registry(config)
    op_profile = build_failure_operational_registry(config)

    assert isinstance(opt_registry, FailureOptimizationProfileRegistry)
    profile = opt_registry.resolve("default_failure_optimization")
    assert profile.profile_id == "default_failure_optimization"
    assert isinstance(profile.controller, FailureOptimizationController)

    assert isinstance(op_profile, FailureOperationalProfile)
    assert op_profile.profile_id == "default_failure_operational"


# --- Determinism & Integration Tests ---


def test_optimization_determinism_and_integration(
    dummy_analysis_input: FailureAnalysisInput,
    dummy_analysis_result: FailureAnalysisResult,
) -> None:
    definition = FailureOptimizationDefinition(timeout_ms=2000.0)
    limiter = BoundedSemaphoreConcurrencyLimiter(max_concurrent_requests=4)
    collector = FailureTelemetryCollector()
    health_monitor = FailureHealthMonitor()

    controller = FailureOptimizationController(
        definition=definition,
        concurrency_limiter=limiter,
        telemetry_collector=collector,
    )

    analyzer = DummyAnalyzer(dummy_analysis_result)

    res1, metrics1 = controller.execute(dummy_analysis_input, analyzer)
    res2, metrics2 = controller.execute(dummy_analysis_input, analyzer)

    # Analytical outputs must be identical
    assert res1 == res2
    assert res1.classification == dummy_analysis_result.classification

    # Telemetry snapshot validation
    snap = collector.snapshot()
    assert snap.total_requests == 2
    assert snap.failure_count == 0

    # Operational health check
    p1 = FailureOptimizationProfile(
        profile_id="p1", definition=definition, controller=controller
    )
    reg = FailureOptimizationProfileRegistry(profiles=(p1,))
    readiness = health_monitor.check_readiness(reg, analyzer_registry="analyzer_reg")
    assert readiness["status"] == "READY"
