"""Unit and integration tests for M4.5 Decision Benchmarking & Evaluation Framework."""

import pytest
from pydantic import ValidationError

from src.core.bootstrap import build_decision_benchmark_registry
from src.core.config import Settings
from src.core.decision import DecisionContext, PolicyDecisionStrategy
from src.core.decision.benchmark import (
    DecisionBenchmarkDataset,
    DecisionBenchmarkExecutor,
    DecisionBenchmarkItem,
    DecisionBenchmarkMetrics,
    DecisionBenchmarkProfile,
    DecisionBenchmarkProfileRegistry,
    DecisionBenchmarkRawOutput,
    DecisionBenchmarkReport,
    DecisionBenchmarkResult,
    DecisionBenchmarkRunner,
    DecisionBenchmarkSuite,
    DecisionMetricEngine,
)
from src.core.exceptions import (
    DecisionBenchmarkProfileNotFoundError,
    DuplicateDecisionBenchmarkProfileError,
)


class MockVerificationResult:
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence


# --- Dataset & Item Tests ---


def test_benchmark_item_validation_and_immutability() -> None:
    context = DecisionContext()
    item = DecisionBenchmarkItem(
        item_id="item_1",
        context=context,
        expected_action="ACCEPT",
    )

    assert item.item_id == "item_1"
    assert item.expected_action == "ACCEPT"

    # Immutability check
    with pytest.raises(ValidationError):
        item.expected_action = "REJECT"


def test_benchmark_dataset_validation_and_immutability() -> None:
    context = DecisionContext()
    item = DecisionBenchmarkItem(
        item_id="item_1",
        context=context,
        expected_action="ACCEPT",
    )
    dataset = DecisionBenchmarkDataset(
        dataset_id="ds_1",
        items=(item,),
    )

    assert dataset.dataset_id == "ds_1"
    assert len(dataset.items) == 1

    # Immutability check
    with pytest.raises(ValidationError):
        dataset.dataset_id = "ds_2"


# --- Suite Configuration Tests ---


def test_benchmark_suite_configuration() -> None:
    context = DecisionContext()
    item = DecisionBenchmarkItem(
        item_id="item_1",
        context=context,
        expected_action="ACCEPT",
    )
    dataset = DecisionBenchmarkDataset(
        dataset_id="ds_1",
        items=(item,),
    )
    suite = DecisionBenchmarkSuite(
        suite_id="suite_1",
        dataset=dataset,
        execution_config={"max_items": 10},
        metadata={"domain": "medical"},
    )

    assert suite.suite_id == "suite_1"
    assert suite.execution_config == {"max_items": 10}
    assert suite.metadata == {"domain": "medical"}


# --- Metrics Engine & Calculators Tests ---


def test_metrics_engine_calculates_all_correctly() -> None:
    # 2 expected matches, 1 abstain, 1 escalate, total 4 items
    # actual: [ACCEPT, ABSTAIN, ESCALATE, REJECT]
    # expected: [ACCEPT, REJECT, ESCALATE, ABSTAIN]
    # matches: expected[0]==actual[0] (ACCEPT), expected[2]==actual[2] (ESCALATE). Total matches = 2 / 4 = 0.50 accuracy
    # abstention rate: 1 / 4 = 0.25
    # escalation rate: 1 / 4 = 0.25
    # latencies: [10.0, 20.0, 30.0, 40.0]. Total time = 100ms. Mean = 25.0ms. Throughput = 4 / 0.1s = 40 QPS.
    raw = DecisionBenchmarkRawOutput(
        suite_id="suite_test",
        item_ids=("1", "2", "3", "4"),
        expected_actions=("ACCEPT", "REJECT", "ESCALATE", "ABSTAIN"),
        actual_actions=("ACCEPT", "ABSTAIN", "ESCALATE", "REJECT"),
        latencies_ms=(10.0, 20.0, 30.0, 40.0),
        decisions=(),
    )

    engine = DecisionMetricEngine()
    metrics = engine.compute(raw)

    assert isinstance(metrics, DecisionBenchmarkMetrics)
    assert metrics.accuracy == 0.50
    assert metrics.abstention_rate == 0.25
    assert metrics.escalation_rate == 0.25
    assert metrics.mean_latency_ms == 25.0
    assert metrics.throughput_qps == pytest.approx(40.0)


# --- Executor Deterministic Execution Tests ---


def test_executor_generates_correct_raw_outputs() -> None:
    # Build dataset
    context1 = DecisionContext(
        verification_result=MockVerificationResult(confidence=0.9)
    )
    context2 = DecisionContext(
        verification_result=MockVerificationResult(confidence=0.1)
    )

    item1 = DecisionBenchmarkItem(
        item_id="item_1",
        context=context1,
        expected_action="ACCEPT",
    )
    item2 = DecisionBenchmarkItem(
        item_id="item_2",
        context=context2,
        expected_action="REJECT",
    )
    dataset = DecisionBenchmarkDataset(
        dataset_id="test_ds",
        items=(item1, item2),
    )

    strategy = PolicyDecisionStrategy()
    executor = DecisionBenchmarkExecutor()
    raw_output = executor.execute(dataset, strategy)

    assert isinstance(raw_output, DecisionBenchmarkRawOutput)
    assert raw_output.suite_id == "test_ds"
    assert raw_output.item_ids == ("item_1", "item_2")
    assert raw_output.expected_actions == ("ACCEPT", "REJECT")

    # Verify actions computed deterministically:
    # item1: confidence 0.9 -> ACCEPT
    # item2: confidence 0.1 -> ABSTAIN (since uncertainty 0.9 > 0.7 max_uncertainty threshold)
    assert raw_output.actual_actions == ("ACCEPT", "ABSTAIN")
    assert len(raw_output.latencies_ms) == 2
    assert all(lat >= 0.0 for lat in raw_output.latencies_ms)


# --- Runner Orchestration Tests ---


def test_runner_report_generation() -> None:
    context = DecisionContext()
    item = DecisionBenchmarkItem(
        item_id="item_1",
        context=context,
        expected_action="REJECT",
    )
    dataset = DecisionBenchmarkDataset(
        dataset_id="ds_1",
        items=(item,),
    )
    suite = DecisionBenchmarkSuite(
        suite_id="suite_1",
        dataset=dataset,
    )
    profile = DecisionBenchmarkProfile(
        profile_id="prof_1",
        enabled_metrics=("accuracy", "abstention_rate"),
        suite_id="suite_1",
    )

    strategy = PolicyDecisionStrategy()
    runner = DecisionBenchmarkRunner(strategy=strategy)
    report = runner.run(suite, profile)

    assert isinstance(report, DecisionBenchmarkReport)
    assert report.suite_id == "suite_1"
    assert report.profile_id == "prof_1"
    assert isinstance(report.result, DecisionBenchmarkResult)
    assert (
        report.result.metrics.accuracy == 1.0
    )  # context has default confidence 0.5 -> REJECT
    assert report.result.metrics.abstention_rate == 0.0


# --- Registry Tests ---


def test_benchmark_registry_duplicate_ids_raises() -> None:
    p1 = DecisionBenchmarkProfile(profile_id="p1", suite_id="s1")
    p2 = DecisionBenchmarkProfile(profile_id="p1", suite_id="s2")  # Duplicate ID

    with pytest.raises(DuplicateDecisionBenchmarkProfileError) as exc_info:
        DecisionBenchmarkProfileRegistry(profiles=(p1, p2))
    assert "Duplicate benchmark profile ID detected: p1" in str(exc_info.value)


def test_benchmark_registry_lookup_success_and_failure() -> None:
    p1 = DecisionBenchmarkProfile(profile_id="p1", suite_id="s1")
    registry = DecisionBenchmarkProfileRegistry(profiles=(p1,))

    # Success lookup
    assert registry.resolve("p1") is p1

    # Failed lookup
    with pytest.raises(DecisionBenchmarkProfileNotFoundError) as exc_info:
        registry.resolve("missing_profile")
    assert "Decision benchmark profile not found: missing_profile" in str(
        exc_info.value
    )


def test_benchmark_registry_compatibility() -> None:
    p1 = DecisionBenchmarkProfile(profile_id="p1", suite_id="s1")
    registry = DecisionBenchmarkProfileRegistry(profiles=(p1,))

    # Compatible
    registry.validate_compatibility("s1")

    # Incompatible
    with pytest.raises(DecisionBenchmarkProfileNotFoundError) as exc_info:
        registry.validate_compatibility("missing_suite_id")
    assert "No benchmark profile configured for suite ID: missing_suite_id" in str(
        exc_info.value
    )


# --- Bootstrap Registry Construction Tests ---


def test_bootstrap_builds_benchmark_registries_correctly() -> None:
    config = Settings()
    registry = build_decision_benchmark_registry(config)

    assert isinstance(registry, DecisionBenchmarkProfileRegistry)
    profile = registry.resolve("default_decision_benchmark")
    assert profile.profile_id == "default_decision_benchmark"
    assert profile.suite_id == "default_decision_suite"


# --- Integration End-to-End Tests ---


def test_decision_benchmark_integration_pipeline() -> None:
    # 1. Setup Dataset
    context1 = DecisionContext(
        verification_result=MockVerificationResult(confidence=0.95)
    )
    context2 = DecisionContext(
        verification_result=MockVerificationResult(confidence=0.05)
    )

    item1 = DecisionBenchmarkItem(
        item_id="item_1",
        context=context1,
        expected_action="ACCEPT",
    )
    item2 = DecisionBenchmarkItem(
        item_id="item_2",
        context=context2,
        expected_action="ABSTAIN",
    )
    dataset = DecisionBenchmarkDataset(
        dataset_id="integration_ds",
        items=(item1, item2),
    )

    # 2. Setup Suite & Profile
    suite = DecisionBenchmarkSuite(
        suite_id="integration_suite",
        dataset=dataset,
    )
    profile = DecisionBenchmarkProfile(
        profile_id="integration_profile",
        enabled_metrics=(
            "accuracy",
            "abstention_rate",
            "escalation_rate",
            "mean_latency_ms",
            "throughput_qps",
        ),
        suite_id="integration_suite",
    )

    # 3. Instantiate Runner
    strategy = PolicyDecisionStrategy()
    runner = DecisionBenchmarkRunner(strategy=strategy)

    # 4. Execute run
    report = runner.run(suite, profile)

    # 5. Verify Report
    assert isinstance(report, DecisionBenchmarkReport)
    assert report.result.metrics.accuracy == 1.0  # Both decisions match expectations!
    assert report.result.metrics.abstention_rate == 0.5
    assert report.result.metrics.escalation_rate == 0.0
    assert report.result.latency_stats["mean_ms"] >= 0.0
    assert report.result.metrics.throughput_qps >= 0.0
