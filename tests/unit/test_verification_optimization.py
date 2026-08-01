"""Unit and integration tests for M2.8 Verification Production Optimization subsystem."""

import time
from typing import Any

import pytest
from pydantic import ValidationError

from src.core.bootstrap import build_verification_optimization_registry
from src.core.calibration.calibration_models import (
    CalibrationResult,
    CalibrationStrategyType,
    CalibrationTrace,
)
from src.core.config import Settings
from src.core.exceptions import (
    DuplicateOptimizationProfileError,
    OptimizationTimeoutError,
)
from src.core.explainability.explanation_models import VerificationExplanationDefinition
from src.core.explainability.implementations import CompositeExplanationStrategy
from src.core.retrieval.retrieval_models import (
    EvidenceBundle,
    EvidencePassage,
    RetrievalMetadata,
)
from src.core.verification.optimization.controller import (
    VerificationOptimizationController,
)
from src.core.verification.optimization.implementations import (
    BoundedSemaphoreVerificationConcurrencyLimiter,
    VerificationTelemetryCollector,
)
from src.core.verification.optimization.optimization_models import (
    OptimizationMode,
    TelemetryLevel,
    VerificationExecutionMetrics,
    VerificationOptimizationDefinition,
    VerificationOptimizationProfile,
    VerificationOptimizationProfileRegistry,
)
from src.core.verification.verification_models import (
    AggregationTrace,
    VerificationResult,
    VerificationVerdict,
    VerifiedPassage,
)


class MockVerifier:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def verify(
        self, claim: str, bundle: EvidenceBundle, definition: Any
    ) -> VerificationResult:
        if self.delay > 0:
            time.sleep(self.delay)
        p1 = EvidencePassage(document_id="d1", span_id="s1", text="text1", score=0.95)
        vp1 = VerifiedPassage(
            passage=p1,
            label=VerificationVerdict.SUPPORTED,
            supports_score=0.9,
            refutes_score=0.05,
            not_enough_info_score=0.05,
        )
        trace = AggregationTrace(
            aggregation_strategy="max_confidence",
            ordered_evaluation_sequence=("s1",),
            weighting_decisions={"s1": 1.0},
            intermediate_scores={},
            final_decision_path="max",
        )
        return VerificationResult(
            verdict=VerificationVerdict.SUPPORTED,
            confidence=0.9,
            supporting_passages=("s1",),
            contradicting_passages=(),
            verified_passages=(vp1,),
            aggregation_trace=trace,
        )


class MockCalibrationStrategy:
    def calibrate(
        self, result: VerificationResult, definition: Any
    ) -> CalibrationResult:
        trace = CalibrationTrace(
            original_confidence=0.9,
            intermediate_values={},
            final_confidence=0.85,
            applied_strategy=CalibrationStrategyType.TEMPERATURE_SCALING,
            parameter_version="1.0",
        )
        return CalibrationResult(
            original_confidence=0.9,
            calibrated_confidence=0.85,
            uncertainty_estimate=0.15,
            calibration_trace=trace,
        )


@pytest.fixture
def dummy_evidence_bundle() -> EvidenceBundle:
    p1 = EvidencePassage(document_id="d1", span_id="s1", text="text1", score=0.95)
    return EvidenceBundle(
        claim="Test claim",
        passages=(p1,),
        metadata=RetrievalMetadata(strategy_id="test", top_k=1),
    )


def test_optimization_models_immutability() -> None:
    defn = VerificationOptimizationDefinition(
        verifier_batch_size=8,
        telemetry_level=TelemetryLevel.DETAILED,
        optimization_mode=OptimizationMode.LATENCY,
    )
    assert defn.verifier_batch_size == 8
    assert defn.telemetry_level == TelemetryLevel.DETAILED
    assert defn.optimization_mode == OptimizationMode.LATENCY

    with pytest.raises(ValidationError):
        # Read-only configuration check
        setattr(defn, "verifier_batch_size", 12)


def test_registry_resolution_and_validation() -> None:
    defn = VerificationOptimizationDefinition()
    prof = VerificationOptimizationProfile(profile_id="p1", definition=defn)

    registry = VerificationOptimizationProfileRegistry(profiles=(prof,))
    assert registry.resolve("p1") is prof

    # Duplicate profile ID detection
    with pytest.raises(DuplicateOptimizationProfileError):
        VerificationOptimizationProfileRegistry(profiles=(prof, prof))


def test_bootstrap_validations_and_fail_fast() -> None:
    settings = Settings()
    registry = build_verification_optimization_registry(settings)
    assert registry.resolve("default_optimization") is not None


def test_concurrency_limiter_semaphore() -> None:
    limiter = BoundedSemaphoreVerificationConcurrencyLimiter(max_concurrency=2)
    assert limiter.max_concurrency == 2
    assert limiter.active_slots == 2

    assert limiter.acquire() is True
    assert limiter.active_slots == 1

    assert limiter.acquire() is True
    assert limiter.active_slots == 0

    # Next attempt should fail immediately
    assert limiter.acquire(timeout_ms=50) is False

    limiter.release()
    assert limiter.active_slots == 1


def test_timeout_enforcement(dummy_evidence_bundle: EvidenceBundle) -> None:
    defn = VerificationOptimizationDefinition(request_timeout_ms=50.0)
    limiter = BoundedSemaphoreVerificationConcurrencyLimiter(max_concurrency=1)
    controller = VerificationOptimizationController(
        definition=defn,
        concurrency_limiter=limiter,
    )

    verifier = MockVerifier(delay=0.1)  # 100ms delay triggers request timeout
    calibrator = MockCalibrationStrategy()
    explainer = CompositeExplanationStrategy()
    exp_defn = VerificationExplanationDefinition(explanation_strategy="COMPOSITE")

    with pytest.raises(OptimizationTimeoutError):
        controller.execute(
            claim="Test claim",
            evidence_bundle=dummy_evidence_bundle,
            verifier=verifier,
            verification_definition=None,
            calibration_strategy=calibrator,
            calibration_definition=None,
            explanation_strategy=explainer,
            explanation_definition=exp_defn,
        )


def test_telemetry_collection_and_snapshots() -> None:
    collector = VerificationTelemetryCollector()
    metrics1 = VerificationExecutionMetrics(
        verification_latency_ms=10.0,
        aggregation_latency_ms=1.0,
        calibration_latency_ms=2.0,
        explanation_latency_ms=3.0,
        total_latency_ms=16.0,
        throughput_qps=62.5,
        memory_usage_bytes=1024,
        batch_sizes={"verifier": 16},
        concurrency_active_requests=1,
    )
    metrics2 = VerificationExecutionMetrics(
        verification_latency_ms=20.0,
        aggregation_latency_ms=1.0,
        calibration_latency_ms=2.0,
        explanation_latency_ms=3.0,
        total_latency_ms=26.0,
        throughput_qps=38.4,
        memory_usage_bytes=1024,
        batch_sizes={"verifier": 16},
        concurrency_active_requests=2,
    )

    collector.record_execution(metrics1)
    collector.record_execution(metrics2)

    snap = collector.snapshot()
    assert snap.total_requests == 2
    assert snap.peak_concurrency == 2
    assert snap.average_latency_ms == 21.0
    assert snap.p95_latency_ms == 26.0
    assert snap.throughput_qps > 0.0

    collector.clear()
    snap_cleared = collector.snapshot()
    assert snap_cleared.total_requests == 0


def test_determinism_and_trace(dummy_evidence_bundle: EvidenceBundle) -> None:
    defn = VerificationOptimizationDefinition(telemetry_enabled=True)
    limiter = BoundedSemaphoreVerificationConcurrencyLimiter(max_concurrency=4)
    collector = VerificationTelemetryCollector()
    controller = VerificationOptimizationController(
        definition=defn,
        concurrency_limiter=limiter,
        telemetry_collector=collector,
        profile_id="optimized_balanced",
    )

    verifier = MockVerifier()
    calibrator = MockCalibrationStrategy()
    explainer = CompositeExplanationStrategy()
    exp_defn = VerificationExplanationDefinition(explanation_strategy="COMPOSITE")

    res1, metrics1, trace1 = controller.execute(
        claim="Test claim",
        evidence_bundle=dummy_evidence_bundle,
        verifier=verifier,
        verification_definition=None,
        calibration_strategy=calibrator,
        calibration_definition=None,
        explanation_strategy=explainer,
        explanation_definition=exp_defn,
    )

    res2, metrics2, trace2 = controller.execute(
        claim="Test claim",
        evidence_bundle=dummy_evidence_bundle,
        verifier=verifier,
        verification_definition=None,
        calibration_strategy=calibrator,
        calibration_definition=None,
        explanation_strategy=explainer,
        explanation_definition=exp_defn,
    )

    # Invariants and Output equivalence (Bit-for-bit identical outputs)
    assert res1.sections == res2.sections
    assert res1.evidence_attribution == res2.evidence_attribution
    assert res1.decision_trace == res2.decision_trace

    # Trace metrics
    assert trace1.profile_id == "optimized_balanced"
    assert trace1.timeout_ms == 5000.0
    assert trace1.telemetry_configured is True
    assert "verifier" in trace1.batch_configuration
