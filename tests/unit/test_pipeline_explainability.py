"""Unit and integration tests for M5.5 Pipeline Explainability & Execution Audit Framework."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.main import create_app
from src.core.bootstrap import build_pipeline_explanation_registry
from src.core.config import Settings
from src.core.evaluation.evaluation_models import (
    EvaluationMetadata,
    EvaluationMetric,
    EvaluationResult,
)
from src.core.exceptions import (
    DuplicatePipelineExplanationProfileError,
    PipelineExplanationConfigurationError,
    PipelineExplanationProfileNotFoundError,
)
from src.core.explainability.explainability_models import (
    ExplanationMetadata,
    ExplanationResult,
    ExplanationSection,
)
from src.core.pipeline.explainability import (
    CompositePipelineExplanationStrategy,
    ExecutionTraceStrategy,
    JsonPipelineRenderer,
    MarkdownPipelineRenderer,
    PipelineAuditReport,
    PipelineDecisionTrace,
    PipelineExecutionExplanation,
    PipelineExplanationDefinition,
    PipelineExplanationEngine,
    PipelineExplanationFormat,
    PipelineExplanationInput,
    PipelineExplanationProfile,
    PipelineExplanationProfileRegistry,
    PipelineExplanationResult,
    PipelineStageExplanation,
    StageBreakdownStrategy,
    SummaryExplanationStrategy,
    TextPipelineRenderer,
    generate_sha256_trace_id,
)
from src.core.pipeline.pipeline_models import (
    PipelineExecutionContext,
    PipelineExecutionResult,
    PipelineRuntimeMetadata,
    PipelineStageMetadata,
)
from src.core.pipeline.resilience import (
    ResilienceRuntimeMetadata,
    RetryAttemptRecord,
    RetryExecutionTrace,
)
from src.core.pipeline.telemetry import (
    PipelineStageAggregation,
    PipelineTelemetrySnapshot,
)


# ==========================================
# Helpers
# ==========================================
def create_dummy_pipeline_execution_result(
    execution_id: str = "exec_123",
    pipeline_id: str = "pipe_abc",
    claim: str = "Arbiter is fully stateless.",
    success: bool = True,
    total_latency_ms: float = 120.5,
    stage_metadata: list[PipelineStageMetadata] | None = None,
) -> PipelineExecutionResult:
    if stage_metadata is None:
        stage_metadata = [
            PipelineStageMetadata(
                stage_id="retrieval",
                profile_id="default_retrieval",
                latency_ms=45.0,
                success=True,
            ),
            PipelineStageMetadata(
                stage_id="verification",
                profile_id="default_verification",
                latency_ms=75.5,
                success=True,
            ),
        ]

    runtime = PipelineRuntimeMetadata(
        pipeline_version="1.0.0",
        configuration_fingerprint="fingerprint_xyz",
        schema_version="1.0.0",
        execution_environment="test",
        execution_timestamp=datetime.now(timezone.utc),
    )

    context = PipelineExecutionContext(
        execution_id=execution_id,
        pipeline_id=pipeline_id,
        claim=claim,
        runtime_metadata=runtime,
        stage_metadata=tuple(stage_metadata),
        total_latency_ms=total_latency_ms,
        success=success,
    )

    eval_result = EvaluationResult(
        metrics=(
            EvaluationMetric(
                identifier="test_metric",
                title="Test Metric",
                score=1.0,
            ),
        ),
        explanation_result=ExplanationResult(
            sections=(
                ExplanationSection(
                    identifier="test_section",
                    title="Test Section",
                    content="Test Content",
                ),
            ),
            decision_result=None,
            metadata=ExplanationMetadata(strategy_id="test_strategy"),
        ),
        metadata=EvaluationMetadata(strategy_id="test_strategy"),
    )

    return PipelineExecutionResult(
        evaluation_result=eval_result,
        execution_context=context,
    )


# ==========================================
# Immutable Models Tests
# ==========================================
def test_models_immutability() -> None:
    definition = PipelineExplanationDefinition(
        template_format=PipelineExplanationFormat.MARKDOWN
    )
    with pytest.raises(ValidationError):
        definition.include_stage_breakdown = False

    stage_exp = PipelineStageExplanation(
        stage_id="retrieval",
        profile_id="default_retrieval",
        latency_ms=45.0,
        success=True,
        trace_id="abc",
    )
    with pytest.raises(ValidationError):
        stage_exp.latency_ms = 50.0


# ==========================================
# Centralized Trace Generation Tests
# ==========================================
def test_deterministic_trace_ids() -> None:
    data1 = "some_string_to_hash"
    data2 = "some_string_to_hash"
    data3 = "another_string_to_hash"

    hash1 = generate_sha256_trace_id(data1)
    hash2 = generate_sha256_trace_id(data2)
    hash3 = generate_sha256_trace_id(data3)

    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 16


# ==========================================
# Strategy Tests
# ==========================================
def test_summary_explanation_strategy() -> None:
    exec_result = create_dummy_pipeline_execution_result()
    input_data = PipelineExplanationInput(execution_result=exec_result)
    definition = PipelineExplanationDefinition()

    strategy = SummaryExplanationStrategy()
    explanation = strategy.generate_explanation(input_data, definition)

    assert explanation.execution_id == "exec_123"
    assert explanation.pipeline_id == "pipe_abc"
    assert explanation.claim_length == len("Arbiter is fully stateless.")
    assert explanation.success is True
    assert explanation.total_latency_ms == 120.5
    assert explanation.summary.outcome == "SUCCESS"
    assert explanation.summary.stage_count == 2
    assert explanation.summary.configuration_fingerprint == "fingerprint_xyz"
    assert explanation.stage_explanations == ()
    assert explanation.decision_trace is None
    assert explanation.metadata["trace_id"] is not None


def test_execution_trace_strategy_no_resilience() -> None:
    exec_result = create_dummy_pipeline_execution_result()
    input_data = PipelineExplanationInput(execution_result=exec_result)
    definition = PipelineExplanationDefinition()

    strategy = ExecutionTraceStrategy()
    explanation = strategy.generate_explanation(input_data, definition)

    assert explanation.decision_trace is None


def test_execution_trace_strategy_with_resilience() -> None:
    exec_result = create_dummy_pipeline_execution_result()

    attempt1 = RetryAttemptRecord(
        attempt_number=1,
        error_type="PipelineStageExecutionError",
        error_message="Stage failed",
        latency_ms=45.0,
    )
    attempt2 = RetryAttemptRecord(
        attempt_number=2,
        error_type="PipelineStageExecutionError",
        error_message="Stage failed again",
        latency_ms=45.0,
    )
    retry_trace = RetryExecutionTrace(
        execution_id="exec_123",
        total_attempts=2,
        succeeded=True,
        attempts=(attempt1, attempt2),
        total_retry_overhead_ms=90.0,
        terminal_error=None,
    )
    res_meta = ResilienceRuntimeMetadata(
        pipeline_profile_id="pipe_abc",
        resilience_profile_id="default_resilience",
        timeout_enforced=True,
        retry_trace=retry_trace,
        recovery_invoked=True,
        recovery_strategy_id="fallback_recovery",
        observed_at=datetime.now(timezone.utc),
    )
    object.__setattr__(exec_result, "resilience_metadata", res_meta)

    input_data = PipelineExplanationInput(execution_result=exec_result)
    definition = PipelineExplanationDefinition()

    strategy = ExecutionTraceStrategy()
    explanation = strategy.generate_explanation(input_data, definition)

    assert explanation.decision_trace is not None
    assert explanation.decision_trace.total_attempts == 2
    assert explanation.decision_trace.succeeded_on_attempt == 2
    assert explanation.decision_trace.timeout_enforced is True
    assert explanation.decision_trace.recovery_invoked is True
    assert explanation.decision_trace.recovery_strategy_id == "fallback_recovery"
    assert explanation.decision_trace.total_retry_overhead_ms == 90.0
    assert explanation.decision_trace.trace_id == generate_sha256_trace_id(
        "resilience_exec_123"
    )


def test_stage_breakdown_strategy_without_telemetry() -> None:
    exec_result = create_dummy_pipeline_execution_result()
    input_data = PipelineExplanationInput(execution_result=exec_result)
    definition = PipelineExplanationDefinition()

    strategy = StageBreakdownStrategy()
    explanation = strategy.generate_explanation(input_data, definition)

    assert len(explanation.stage_explanations) == 2
    assert explanation.stage_explanations[0].stage_id == "retrieval"
    assert explanation.stage_explanations[0].latency_ms == 45.0
    assert explanation.stage_explanations[0].success is True
    assert explanation.stage_explanations[0].latency_percentile_rank is None
    assert (
        explanation.stage_explanations[0].observation
        == "Stage 'retrieval' completed in 45.00ms (success)"
    )


def test_stage_breakdown_strategy_with_telemetry() -> None:
    exec_result = create_dummy_pipeline_execution_result()

    agg1 = PipelineStageAggregation(
        stage_id="retrieval",
        profile_id="default_retrieval",
        execution_count=10,
        success_count=10,
        failure_count=0,
        mean_latency_ms=10.0,
        p50_latency_ms=10.0,
        p90_latency_ms=10.0,
        p99_latency_ms=10.0,
        success_rate=1.0,
    )
    agg2 = PipelineStageAggregation(
        stage_id="verification",
        profile_id="default_verification",
        execution_count=10,
        success_count=10,
        failure_count=0,
        mean_latency_ms=50.0,
        p50_latency_ms=50.0,
        p90_latency_ms=50.0,
        p99_latency_ms=50.0,
        success_rate=1.0,
    )
    agg3 = PipelineStageAggregation(
        stage_id="decision",
        profile_id="default_decision",
        execution_count=10,
        success_count=10,
        failure_count=0,
        mean_latency_ms=80.0,
        p50_latency_ms=80.0,
        p90_latency_ms=80.0,
        p99_latency_ms=80.0,
        success_rate=1.0,
    )

    telemetry = PipelineTelemetrySnapshot(
        pipeline_id="pipe_abc",
        total_executions=100,
        successful_executions=95,
        failed_executions=5,
        mean_total_latency_ms=110.0,
        p50_total_latency_ms=110.0,
        p90_total_latency_ms=110.0,
        p99_total_latency_ms=110.0,
        overall_success_rate=0.95,
        stage_aggregations=(agg1, agg2, agg3),
        snapshot_timestamp=datetime.now(timezone.utc),
    )

    input_data = PipelineExplanationInput(
        execution_result=exec_result, telemetry_snapshot=telemetry
    )
    definition = PipelineExplanationDefinition()

    strategy = StageBreakdownStrategy()
    explanation = strategy.generate_explanation(input_data, definition)

    # Retrieval actual latency = 45.0. Telemetry mean latencies = 10.0, 50.0, 80.0.
    # Lower than 45.0 is only 10.0 (1 out of 3 -> 33.33%).
    assert explanation.stage_explanations[0].latency_percentile_rank == pytest.approx(
        33.333, abs=0.01
    )
    assert explanation.telemetry_context.total_executions == 100
    assert explanation.telemetry_context.overall_success_rate == 0.95


def test_composite_pipeline_explanation_strategy() -> None:
    exec_result = create_dummy_pipeline_execution_result()
    input_data = PipelineExplanationInput(execution_result=exec_result)
    definition = PipelineExplanationDefinition()

    strategy = CompositePipelineExplanationStrategy()
    explanation = strategy.generate_explanation(input_data, definition)

    assert explanation.summary.outcome == "SUCCESS"
    assert len(explanation.stage_explanations) == 2
    assert explanation.decision_trace is None
    assert "trace_id" in explanation.metadata


# ==========================================
# Registry Tests
# ==========================================
def test_registry_duplicate_prevention() -> None:
    definition = PipelineExplanationDefinition()
    strategy = SummaryExplanationStrategy()

    profile1 = PipelineExplanationProfile(
        profile_id="profile_1", definition=definition, strategy=strategy
    )
    profile2 = PipelineExplanationProfile(
        profile_id="profile_1", definition=definition, strategy=strategy
    )

    with pytest.raises(DuplicatePipelineExplanationProfileError):
        PipelineExplanationProfileRegistry(profiles=(profile1, profile2))


def test_registry_resolution() -> None:
    definition = PipelineExplanationDefinition()
    strategy = SummaryExplanationStrategy()

    profile = PipelineExplanationProfile(
        profile_id="profile_1", definition=definition, strategy=strategy
    )
    registry = PipelineExplanationProfileRegistry(profiles=(profile,))

    resolved = registry.resolve("profile_1")
    assert resolved.profile_id == "profile_1"

    with pytest.raises(PipelineExplanationProfileNotFoundError):
        registry.resolve("non_existent")


# ==========================================
# Engine Tests
# ==========================================
def test_pipeline_explanation_engine() -> None:
    definition = PipelineExplanationDefinition(
        template_format=PipelineExplanationFormat.MARKDOWN
    )
    strategy = CompositePipelineExplanationStrategy()
    profile = PipelineExplanationProfile(
        profile_id="profile_composite", definition=definition, strategy=strategy
    )
    registry = PipelineExplanationProfileRegistry(profiles=(profile,))

    engine = PipelineExplanationEngine(registry=registry)

    exec_result = create_dummy_pipeline_execution_result()
    input_data = PipelineExplanationInput(execution_result=exec_result)

    result = engine.explain(input_data, "profile_composite")

    assert isinstance(result, PipelineExplanationResult)
    assert result.renderer_id == "markdown"
    assert result.strategy_id == "pipeline_composite"
    assert "rendered_content" in result.metadata
    assert "# Pipeline Execution Audit Report" in result.metadata["rendered_content"]

    res, report = engine.explain_with_audit(input_data, "profile_composite")
    assert isinstance(res, PipelineExplanationResult)
    assert isinstance(report, PipelineAuditReport)
    assert report.execution_id == "exec_123"
    assert report.profile_id == "profile_composite"


# ==========================================
# Renderer Tests
# ==========================================
def test_renderers_correctness() -> None:
    exec_result = create_dummy_pipeline_execution_result()
    input_data = PipelineExplanationInput(execution_result=exec_result)
    definition = PipelineExplanationDefinition()

    strategy = CompositePipelineExplanationStrategy()
    explanation = strategy.generate_explanation(input_data, definition)

    markdown_renderer = MarkdownPipelineRenderer()
    md_output = markdown_renderer.render(explanation)
    assert "# Pipeline Execution Audit Report" in md_output
    assert "## Execution Summary" in md_output
    assert "retrieval" in md_output

    json_renderer = JsonPipelineRenderer()
    json_output = json_renderer.render(explanation)
    assert '"execution_id": "exec_123"' in json_output

    text_renderer = TextPipelineRenderer()
    text_output = text_renderer.render(explanation)
    assert "PIPELINE EXECUTION AUDIT REPORT" in text_output
    assert "STAGE BREAKDOWN:" in text_output


# ==========================================
# Bootstrap / DI Tests
# ==========================================
def test_bootstrap_integration() -> None:
    settings = Settings()
    registry = build_pipeline_explanation_registry(settings)
    assert isinstance(registry, PipelineExplanationProfileRegistry)

    resolved = registry.resolve("default_pipeline_explanation")
    assert resolved.profile_id == "default_pipeline_explanation"


# ==========================================
# API / Lifespan Integration Tests
# ==========================================
def test_api_lifespan_integration() -> None:
    app = create_app()
    with TestClient(app):
        assert hasattr(app.state, "pipeline_explanation_registry")
        assert app.state.pipeline_explanation_registry is not None
        registry = app.state.pipeline_explanation_registry
        assert isinstance(registry, PipelineExplanationProfileRegistry)
