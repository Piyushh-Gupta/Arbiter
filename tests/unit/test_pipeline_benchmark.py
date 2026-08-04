"""Unit and integration tests for M5.4 Pipeline Benchmarking & Evaluation Framework."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.main import create_app
from src.core.bootstrap import build_pipeline_benchmark_registry
from src.core.config import Settings
from src.core.evaluation.evaluation_models import (
    EvaluationMetadata,
    EvaluationMetric,
    EvaluationResult,
)
from src.core.exceptions import (
    DuplicatePipelineBenchmarkProfileError,
    PipelineBenchmarkConfigurationError,
    PipelineBenchmarkProfileNotFoundError,
)
from src.core.explainability.explainability_models import (
    ExplanationMetadata,
    ExplanationResult,
    ExplanationSection,
)
from src.core.pipeline.benchmark import (
    BaseClock,
    PipelineBenchmarkDataset,
    PipelineBenchmarkDefinition,
    PipelineBenchmarkExecutor,
    PipelineBenchmarkItem,
    PipelineBenchmarkMetric,
    PipelineBenchmarkProfile,
    PipelineBenchmarkProfileRegistry,
    PipelineBenchmarkRawOutput,
    PipelineBenchmarkReport,
    PipelineBenchmarkRunner,
    PipelineBenchmarkSuite,
    PipelineFailureRecord,
)
from src.core.pipeline.benchmark.runner import PipelineBenchmarkReportBuilder
from src.core.pipeline.pipeline_models import (
    PipelineExecutionContext,
    PipelineExecutionResult,
    PipelineRuntimeMetadata,
    PipelineStageMetadata,
)


# ==========================================
# Mock Clock for deterministic latency tests
# ==========================================
class MockClock(BaseClock):
    """Test clock yielding pre-defined sequence of millisecond timestamps."""

    def __init__(self, times: list[float]) -> None:
        self._times = times
        self._idx = 0

    def now_ms(self) -> float:
        val = self._times[self._idx]
        if self._idx < len(self._times) - 1:
            self._idx += 1
        return val


# ==========================================
# Helper to construct dummy execution results
# ==========================================
def create_dummy_execution_result(
    claim: str = "A claim",
    pipeline_profile_id: str = "pipe_1",
    success: bool = True,
    total_latency_ms: float = 100.0,
    stage_latencies: dict[str, float] | None = None,
) -> PipelineExecutionResult:
    metric = EvaluationMetric(
        identifier="dummy_metric",
        title="Dummy Metric",
        score=1.0,
    )
    eval_meta = EvaluationMetadata(
        strategy_id="dummy_strategy",
    )
    exp_section = ExplanationSection(
        identifier="dummy_section",
        title="Dummy Section",
        content="Dummy Content",
    )
    exp_meta = ExplanationMetadata(
        strategy_id="dummy_strategy",
    )
    explanation = ExplanationResult(
        sections=(exp_section,),
        decision_result=None,
        metadata=exp_meta,
    )
    eval_result = EvaluationResult(
        metrics=(metric,),
        explanation_result=explanation,
        metadata=eval_meta,
    )
    runtime_meta = PipelineRuntimeMetadata(
        pipeline_version="1.0.0",
        configuration_fingerprint="fp",
        schema_version="1.0.0",
        execution_environment="test",
        execution_timestamp=datetime.now(timezone.utc),
    )

    stage_latencies = stage_latencies or {
        "retrieval": 20.0,
        "verification": 30.0,
        "failure_analysis": 10.0,
        "uncertainty": 5.0,
        "decision": 15.0,
        "explanation": 10.0,
        "evaluation": 10.0,
    }

    stage_meta = tuple(
        PipelineStageMetadata(
            stage_id=stage_id,
            profile_id="default",
            latency_ms=lat,
            success=True,
        )
        for stage_id, lat in stage_latencies.items()
    )

    ctx = PipelineExecutionContext(
        execution_id="exec_1",
        pipeline_id=pipeline_profile_id,
        claim=claim,
        runtime_metadata=runtime_meta,
        stage_metadata=stage_meta,
        total_latency_ms=total_latency_ms,
        success=success,
    )
    return PipelineExecutionResult(
        evaluation_result=eval_result,
        execution_context=ctx,
    )


# ==========================================
# 1. Immutability Tests
# ==========================================
def test_models_immutability() -> None:
    item = PipelineBenchmarkItem(
        item_id="item_1",
        claim="Claim 1",
        pipeline_profile_id="pipe_1",
    )
    with pytest.raises(ValidationError):
        setattr(item, "claim", "Claim 2")

    dataset = PipelineBenchmarkDataset(
        dataset_id="ds_1",
        items=(item,),
    )
    with pytest.raises(ValidationError):
        setattr(dataset, "dataset_id", "ds_2")

    definition = PipelineBenchmarkDefinition()
    with pytest.raises(ValidationError):
        setattr(definition, "include_stage_breakdown", False)


# ==========================================
# 2. Registry Validation & Compatibility
# ==========================================
def test_profile_registry_validation() -> None:
    definition = PipelineBenchmarkDefinition()
    p1 = PipelineBenchmarkProfile(
        profile_id="p1",
        suite_id="suite_a",
        definition=definition,
    )
    p2 = PipelineBenchmarkProfile(
        profile_id="p2",
        suite_id="suite_b",
        definition=definition,
    )

    # Valid registry
    registry = PipelineBenchmarkProfileRegistry(profiles=(p1, p2))
    assert registry.resolve("p1") == p1
    assert registry.resolve("p2") == p2

    # Duplicate detection
    with pytest.raises(DuplicatePipelineBenchmarkProfileError):
        PipelineBenchmarkProfileRegistry(profiles=(p1, p1))

    # Profile not found
    with pytest.raises(PipelineBenchmarkProfileNotFoundError):
        registry.resolve("non_existent")

    # Compatibility validation
    registry.validate_compatibility("suite_a")
    registry.validate_compatibility("suite_b")
    with pytest.raises(PipelineBenchmarkProfileNotFoundError):
        registry.validate_compatibility("suite_c")


# ==========================================
# 3. Metric Correctness & Percentiles
# ==========================================
def test_percentile_calculations() -> None:
    from src.core.pipeline.benchmark.metrics import _percentile

    # Empty list
    assert _percentile((), 50.0) == 0.0

    # Single item
    assert _percentile((42.0,), 99.0) == 42.0

    # Odd dataset: [10, 20, 30] sorted.
    # p=50.0 -> idx = floor(0.5 * 3) = 1 -> 20.0
    # p=95.0 -> idx = floor(0.95 * 3) = 2 -> 30.0
    latencies = (30.0, 10.0, 20.0)
    assert _percentile(latencies, 50.0) == 20.0
    assert _percentile(latencies, 95.0) == 30.0
    assert _percentile(latencies, 99.0) == 30.0

    # Even dataset: [10, 20, 30, 40] sorted.
    # p=50.0 -> idx = floor(0.5 * 4) = 2 -> 30.0
    # p=25.0 -> idx = floor(0.25 * 4) = 1 -> 20.0
    latencies_even = (40.0, 20.0, 10.0, 30.0)
    assert _percentile(latencies_even, 50.0) == 30.0
    assert _percentile(latencies_even, 25.0) == 20.0


def test_determinism_rate_metric() -> None:
    from src.core.pipeline.benchmark.metrics import PipelineDeterminismRateCalculator

    calc = PipelineDeterminismRateCalculator()

    # Empty dataset
    raw_empty = PipelineBenchmarkRawOutput(suite_id="test")
    assert calc.calculate(raw_empty) == 1.0

    # All unique claims
    raw_unique = PipelineBenchmarkRawOutput(
        suite_id="test",
        claims=("c1", "c2", "c3"),
        actual_successes=(True, False, True),
    )
    assert calc.calculate(raw_unique) == 1.0

    # Repeated claims with deterministic outcomes
    raw_deterministic = PipelineBenchmarkRawOutput(
        suite_id="test",
        claims=("c1", "c1", "c2", "c2"),
        actual_successes=(True, True, False, False),
    )
    assert calc.calculate(raw_deterministic) == 1.0

    # Repeated claims with non-deterministic outcomes
    raw_nondeterministic = PipelineBenchmarkRawOutput(
        suite_id="test",
        claims=("c1", "c1", "c2", "c2"),
        actual_successes=(True, False, False, False),
    )
    # c1 has mixed outcomes, c2 has same outcomes -> 1 out of 2 is deterministic
    assert calc.calculate(raw_nondeterministic) == 0.5


def test_metric_engine_execution() -> None:
    from src.core.pipeline.benchmark.metrics import PipelineBenchmarkMetricEngine

    raw_output = PipelineBenchmarkRawOutput(
        suite_id="suite_1",
        item_ids=("item_1", "item_2"),
        claims=("claim_1", "claim_2"),
        expected_successes=(True, True),
        actual_successes=(True, False),
        total_latencies_ms=(100.0, 200.0),
        stage_latencies_ms=(
            {"stage_a": 40.0, "stage_b": 60.0},
            {"stage_a": 50.0, "stage_b": 150.0},
        ),
        retry_attempt_counts=(1, 3),
        timeout_triggered=(False, True),
        recovery_invoked=(False, True),
    )

    engine = PipelineBenchmarkMetricEngine()
    metrics = engine.compute(raw_output)

    assert metrics.success_rate == 0.5
    assert metrics.mean_latency_ms == 150.0
    assert (
        metrics.p50_latency_ms == 200.0
    )  # sorted: [100.0, 200.0]. floor(0.5*2)=1 -> 200.0
    assert metrics.throughput_qps == 2.0 / 0.3  # 2 items / 0.3 seconds
    assert metrics.retry_rate == 0.5  # item 2 retried (attempts > 1)
    assert metrics.mean_retry_attempts == 2.0
    assert metrics.timeout_rate == 0.5
    assert metrics.recovery_rate == 0.5
    assert metrics.determinism_rate == 1.0

    # Stage metrics breakdown
    stage_metrics = engine.compute_stage_metrics(raw_output)
    assert "stage_a" in stage_metrics
    assert "stage_b" in stage_metrics
    assert stage_metrics["stage_a"].mean_latency_ms == 45.0
    assert stage_metrics["stage_b"].mean_latency_ms == 105.0


# ==========================================
# 4. Executor & Runner Tests
# ==========================================
def test_executor_and_failure_capture() -> None:
    clock = MockClock([1000.0, 1150.0, 2000.0, 2300.0])
    executor = PipelineBenchmarkExecutor(clock=clock)

    # Item 1 succeeds, Item 2 raises an exception
    res1 = create_dummy_execution_result(
        claim="claim 1",
        pipeline_profile_id="pipe_1",
        success=True,
        total_latency_ms=120.0,
    )
    # Mock resilience metadata on res1
    mock_trace = MagicMock()
    mock_trace.total_attempts = 1
    mock_trace.attempts = ()
    mock_trace.terminal_error = None
    mock_res_meta = MagicMock()
    mock_res_meta.retry_trace = mock_trace
    mock_res_meta.recovery_invoked = False
    object.__setattr__(res1, "resilience_metadata", mock_res_meta)

    # Exception for Item 2
    from src.core.exceptions import PipelineStageExecutionError

    exc = PipelineStageExecutionError("Stage retrieval failed!")
    mock_exc_trace = MagicMock()
    mock_exc_trace.total_attempts = 3
    mock_exc_trace.attempts = (MagicMock(error_type="PipelineResilienceTimeoutError"),)
    mock_exc_trace.terminal_error = "Timeout occurred"
    mock_exc_res_meta = MagicMock()
    mock_exc_res_meta.retry_trace = mock_exc_trace
    mock_exc_res_meta.recovery_invoked = True
    object.__setattr__(exc, "resilience_metadata", mock_exc_res_meta)

    orchestrator = MagicMock()
    orchestrator.execute.side_effect = [res1, exc]

    dataset = PipelineBenchmarkDataset(
        dataset_id="ds_1",
        items=(
            PipelineBenchmarkItem(
                item_id="i1", claim="c1", pipeline_profile_id="pipe_1"
            ),
            PipelineBenchmarkItem(
                item_id="i2", claim="c2", pipeline_profile_id="pipe_1"
            ),
        ),
    )

    raw_output = executor.execute(dataset, orchestrator)

    assert raw_output.suite_id == "ds_1"
    assert raw_output.item_ids == ("i1", "i2")
    assert raw_output.actual_successes == (True, False)
    # Item 1 uses its internal execution context total_latency_ms (120.0)
    # Item 2 uses elapsed clock latency: 2300 - 2000 = 300.0
    assert raw_output.total_latencies_ms == (120.0, 300.0)
    assert raw_output.retry_attempt_counts == (1, 3)
    assert raw_output.timeout_triggered == (False, True)
    assert raw_output.recovery_invoked == (False, True)

    # Failures mapping
    assert raw_output.failures[0] is None
    assert isinstance(raw_output.failures[1], PipelineFailureRecord)
    assert raw_output.failures[1].exception_type == "PipelineStageExecutionError"
    assert raw_output.failures[1].error_message == "Stage retrieval failed!"
    assert raw_output.failures[1].failure_category == "STAGE_ERROR"
    assert raw_output.failures[1].retry_attempts == 3


def test_runner_execution() -> None:
    res1 = create_dummy_execution_result(claim="c1")
    orchestrator = MagicMock()
    orchestrator.execute.return_value = res1

    runner = PipelineBenchmarkRunner(orchestrator=orchestrator)

    dataset = PipelineBenchmarkDataset(
        dataset_id="ds_test",
        items=(
            PipelineBenchmarkItem(
                item_id="i1", claim="c1", pipeline_profile_id="pipe_1"
            ),
        ),
    )
    suite = PipelineBenchmarkSuite(suite_id="suite_test", dataset=dataset)
    definition = PipelineBenchmarkDefinition(
        enabled_metrics=(PipelineBenchmarkMetric.SUCCESS_RATE,),
        include_stage_breakdown=True,
    )
    profile = PipelineBenchmarkProfile(
        profile_id="prof_test",
        suite_id="suite_test",
        definition=definition,
    )

    report = runner.run(suite, profile)

    assert isinstance(report, PipelineBenchmarkReport)
    assert report.suite_id == "suite_test"
    assert report.profile_id == "prof_test"
    assert report.pipeline_profile_id == "pipe_1"
    assert report.result.item_count == 1
    assert report.result.metrics.success_rate == 1.0
    assert "retrieval" in report.result.stage_metrics


def test_report_builder_latency_statistics() -> None:
    # Single item
    stats = PipelineBenchmarkReportBuilder.calculate_latency_stats([100.0])
    assert stats["min_ms"] == 100.0
    assert stats["max_ms"] == 100.0
    assert stats["mean_ms"] == 100.0
    assert stats["stddev_ms"] == 0.0

    # Multi item
    stats_multi = PipelineBenchmarkReportBuilder.calculate_latency_stats([100.0, 200.0])
    assert stats_multi["min_ms"] == 100.0
    assert stats_multi["max_ms"] == 200.0
    assert stats_multi["mean_ms"] == 150.0
    assert stats_multi["stddev_ms"] > 0.0


# ==========================================
# 5. Bootstrap Integration
# ==========================================
def test_bootstrap_integration() -> None:
    config = Settings()

    # Valid config resolution
    registry = build_pipeline_benchmark_registry(config)
    assert isinstance(registry, PipelineBenchmarkProfileRegistry)
    profile = registry.resolve(config.pipeline_benchmark.active_profile_id)
    assert profile.suite_id == config.pipeline_benchmark.default_suite_id

    # Unknown metric configuration raises
    config.pipeline_benchmark.enabled_metrics.append("invalid_metric_name")
    with pytest.raises(PipelineBenchmarkConfigurationError):
        build_pipeline_benchmark_registry(config)


# ==========================================
# 6. API & State Integration
# ==========================================
def test_api_lifespan_integration() -> None:
    app = create_app()
    with TestClient(app):
        # Access application state
        assert hasattr(app.state, "pipeline_benchmark_registry")
        registry = app.state.pipeline_benchmark_registry
        assert isinstance(registry, PipelineBenchmarkProfileRegistry)
        assert registry.resolve("default_pipeline_benchmark") is not None
