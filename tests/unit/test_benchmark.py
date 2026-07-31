"""Unit and integration tests for the benchmarking and evaluation framework."""

import math
import time
from typing import Any

import pytest

from src.core.benchmark.base import BaseBenchmarkDataset
from src.core.benchmark.benchmark_models import BenchmarkDefinition, BenchmarkMetricType
from src.core.benchmark.implementations import (
    ClimateFEVERDataset,
    FEVERDataset,
    LocalBenchmarkDataset,
    SciFactDataset,
)
from src.core.benchmark.metrics import (
    AbstentionRateCalculator,
    AccuracyCalculator,
    BrierScoreCalculator,
    ConflictRateCalculator,
    ECECalculator,
    F1Calculator,
    LowConfidenceRateCalculator,
    MCECalculator,
    MeanLatencyCalculator,
    MicroF1Calculator,
    NegativeLogLikelihoodCalculator,
    P95LatencyCalculator,
    P99LatencyCalculator,
    PrecisionCalculator,
    RecallCalculator,
    ThroughputCalculator,
)
from src.core.benchmark.runner import VerificationBenchmarkRunner
from src.core.bootstrap import build_verification_benchmark_registry
from src.core.config import Settings
from src.core.retrieval.retrieval_models import (
    EvidenceBundle,
    EvidencePassage,
    RetrievalMetadata,
)
from src.core.verification.verification_models import (
    ConflictAnalysis,
    VerificationResult,
    VerificationVerdict,
)


@pytest.fixture
def dummy_verification_results() -> tuple[VerificationResult, ...]:
    vr1 = VerificationResult(
        verdict=VerificationVerdict.SUPPORTED,
        confidence=0.9,
        supporting_passages=("s1",),
        contradicting_passages=(),
        conflict_analysis=ConflictAnalysis(conflict_severity=0.1),
    )
    vr2 = VerificationResult(
        verdict=VerificationVerdict.SUPPORTED,
        confidence=0.8,
        supporting_passages=("s2",),
        contradicting_passages=(),
        conflict_analysis=ConflictAnalysis(conflict_severity=0.2),
    )
    vr3 = VerificationResult(
        verdict=VerificationVerdict.CONTRADICTED,
        confidence=0.4,
        supporting_passages=(),
        contradicting_passages=("s3",),
        conflict_analysis=ConflictAnalysis(conflict_severity=0.6),
    )
    vr4 = VerificationResult(
        verdict=VerificationVerdict.INSUFFICIENT,
        confidence=0.5,
        supporting_passages=(),
        contradicting_passages=(),
        conflict_analysis=ConflictAnalysis(conflict_severity=0.0),
    )
    return (vr1, vr2, vr3, vr4)


@pytest.fixture
def dummy_ground_truths() -> tuple[VerificationVerdict, ...]:
    return (
        VerificationVerdict.SUPPORTED,
        VerificationVerdict.CONTRADICTED,
        VerificationVerdict.CONTRADICTED,
        VerificationVerdict.INSUFFICIENT,
    )


def test_classification_metrics(
    dummy_verification_results: tuple[VerificationResult, ...],
    dummy_ground_truths: tuple[VerificationVerdict, ...],
) -> None:
    acc = AccuracyCalculator().compute(dummy_verification_results, dummy_ground_truths)
    assert acc.value == 0.75

    prec = PrecisionCalculator().compute(
        dummy_verification_results, dummy_ground_truths
    )
    assert pytest.approx(prec.value) == 0.83333333

    rec = RecallCalculator().compute(dummy_verification_results, dummy_ground_truths)
    assert pytest.approx(rec.value) == 0.83333333

    f1 = F1Calculator().compute(dummy_verification_results, dummy_ground_truths)
    assert pytest.approx(f1.value) == 7.0 / 9.0

    micro_f1 = MicroF1Calculator().compute(
        dummy_verification_results, dummy_ground_truths
    )
    assert micro_f1.value == 0.75


def test_calibration_metrics(
    dummy_verification_results: tuple[VerificationResult, ...],
    dummy_ground_truths: tuple[VerificationVerdict, ...],
) -> None:
    ece = ECECalculator().compute(dummy_verification_results, dummy_ground_truths)
    assert pytest.approx(ece.value) == 0.5

    mce = MCECalculator().compute(dummy_verification_results, dummy_ground_truths)
    assert mce.value == 0.8

    brier = BrierScoreCalculator().compute(
        dummy_verification_results, dummy_ground_truths
    )
    assert pytest.approx(brier.value) == 0.405

    nll = NegativeLogLikelihoodCalculator().compute(
        dummy_verification_results, dummy_ground_truths
    )
    expected_nll = (
        -math.log(0.9) - math.log(1e-15) - math.log(0.4) - math.log(0.5)
    ) / 4.0
    assert pytest.approx(nll.value) == expected_nll


def test_performance_metrics() -> None:
    latencies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    mean_lat = MeanLatencyCalculator().compute(latencies, [])
    assert mean_lat.value == pytest.approx(0.55)

    p95 = P95LatencyCalculator().compute(latencies, [])
    assert p95.value == 1.0

    p99 = P99LatencyCalculator().compute(latencies, [])
    assert p99.value == 1.0

    tput = ThroughputCalculator().compute(latencies, [])
    assert pytest.approx(tput.value) == 10.0 / 5.5


def test_robustness_metrics(
    dummy_verification_results: tuple[VerificationResult, ...],
) -> None:
    abst = AbstentionRateCalculator().compute(dummy_verification_results, [])
    assert abst.value == 0.25

    low_conf = LowConfidenceRateCalculator(threshold=0.5).compute(
        dummy_verification_results, []
    )
    assert low_conf.value == 0.25

    conflict = ConflictRateCalculator(threshold=0.3).compute(
        dummy_verification_results, []
    )
    assert conflict.value == 0.25


def test_local_benchmark_dataset() -> None:
    samples = ({"claim": "c1", "bundle": "b1", "ground_truth_verdict": "SUPPORTED"},)
    dataset = LocalBenchmarkDataset(
        dataset_name="test_ds",
        samples=samples,
        version="2.0",
        metadata={"desc": "test"},
    )
    assert dataset.load_samples() == samples
    assert dataset.dataset_version() == "2.0"
    assert dataset.dataset_metadata()["desc"] == "test"

    assert FEVERDataset(samples).dataset_version() == "1.0"
    assert SciFactDataset(samples).dataset_version() == "1.0"
    assert ClimateFEVERDataset(samples).dataset_version() == "1.0"


class MockVerifierProfile:
    def __init__(self) -> None:
        from src.core.verification.verification_models import VerificationDefinition

        self.verifier = self
        self.definition = VerificationDefinition(
            verifier_model="mock-nli",
            aggregation_strategy="max_confidence",
        )
        self.verifier_model = "mock-nli"
        self.aggregation_strategy = self.definition.aggregation_strategy
        self.confidence_thresholds: dict[str, float] = {}

    def verify_passages(self, claim: str, bundle: Any) -> tuple[Any, ...]:
        from src.core.verification.verification_models import (
            PassageVerificationResult,
            PassageVerificationScore,
        )

        return (
            PassageVerificationResult(
                span_id="s1",
                verdict=VerificationVerdict.SUPPORTED,
                confidence=0.8,
                probability_distribution=PassageVerificationScore(
                    entailment_probability=0.8,
                    contradiction_probability=0.1,
                    neutral_probability=0.1,
                ),
            ),
        )

    def aggregate(self, claim_input: Any, passage_results: Any) -> VerificationResult:
        from typing import cast

        return cast(VerificationResult, passage_results[0])


class MockCalibrationProfile:
    def __init__(self) -> None:
        from src.core.calibration.calibration_models import (
            CalibrationDefinition,
            CalibrationStrategyType,
        )

        self.strategy = self
        self.definition = CalibrationDefinition(
            strategy=CalibrationStrategyType.IDENTITY,
            parameters=None,
        )
        self.uncertainty_estimator = self

    def calibrate(self, result: Any, definition: Any, estimator: Any) -> Any:
        return result


class MockVerificationRegistry:
    def resolve(self, profile_id: str) -> Any:
        return MockVerifierProfile()


class MockCalibrationRegistry:
    def resolve(self, profile_id: str) -> Any:
        return MockCalibrationProfile()


def test_verification_benchmark_runner() -> None:
    ver_reg = MockVerificationRegistry()
    cal_reg = MockCalibrationRegistry()

    p1 = EvidencePassage(document_id="d1", span_id="s1", text="text", score=0.8)
    bundle = EvidenceBundle(
        claim="c1",
        passages=(p1,),
        metadata=RetrievalMetadata(strategy_id="test", top_k=1),
    )
    samples = (
        {
            "claim": "c1",
            "bundle": bundle,
            "ground_truth_verdict": "SUPPORTED",
            "sample_id": "sample_1",
        },
    )
    dataset = LocalBenchmarkDataset("FEVER", samples)
    datasets: dict[str, BaseBenchmarkDataset] = {"FEVER": dataset}

    runner = VerificationBenchmarkRunner(ver_reg, cal_reg, datasets)

    definition = BenchmarkDefinition(
        benchmark_name="test_run",
        dataset_identifier="FEVER",
        selected_metrics=(
            BenchmarkMetricType.ACCURACY,
            BenchmarkMetricType.MEAN_LATENCY,
        ),
        evaluation_profile_id="default_verification",
    )

    report = runner.run_benchmark(definition)
    assert report.benchmark_result.confusion_matrix["SUPPORTED"]["SUPPORTED"] == 1
    assert len(report.benchmark_result.metrics) == 2
    assert report.benchmark_trace.dataset_version == "1.0"
    assert report.benchmark_trace.execution_sequence == ("sample_1",)
    assert report.configuration_fingerprint is not None


def test_bootstrap_registry() -> None:
    settings = Settings()
    registry = build_verification_benchmark_registry(settings)
    assert registry.resolve("default_benchmark") is not None


def test_determinism() -> None:
    ver_reg = MockVerificationRegistry()
    cal_reg = MockCalibrationRegistry()
    p1 = EvidencePassage(document_id="d1", span_id="s1", text="text", score=0.8)
    bundle = EvidenceBundle(
        claim="c1",
        passages=(p1,),
        metadata=RetrievalMetadata(strategy_id="test", top_k=1),
    )
    samples = (
        {
            "claim": "c1",
            "bundle": bundle,
            "ground_truth_verdict": "SUPPORTED",
            "sample_id": "sample_1",
        },
    )
    dataset = LocalBenchmarkDataset("FEVER", samples)
    datasets: dict[str, BaseBenchmarkDataset] = {"FEVER": dataset}
    runner = VerificationBenchmarkRunner(ver_reg, cal_reg, datasets)

    definition = BenchmarkDefinition(
        benchmark_name="test_run",
        dataset_identifier="FEVER",
        selected_metrics=(BenchmarkMetricType.ACCURACY,),
        evaluation_profile_id="default_verification",
    )

    report1 = runner.run_benchmark(definition)
    time.sleep(0.1)
    report2 = runner.run_benchmark(definition)

    assert (
        report1.benchmark_result.metrics[0].value
        == report2.benchmark_result.metrics[0].value
    )
    assert report1.configuration_fingerprint == report2.configuration_fingerprint
