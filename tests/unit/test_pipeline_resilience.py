"""Unit and integration tests for M5.3 Pipeline Recovery, Retry & Resilience subsystem."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.main import create_app
from src.core.bootstrap import build_resilience_controller, build_resilience_registry
from src.core.config import Settings
from src.core.evaluation.evaluation_models import (
    EvaluationMetadata,
    EvaluationMetric,
    EvaluationResult,
)
from src.core.exceptions import (
    DuplicateResilienceProfileError,
    PipelineResilienceTimeoutError,
    PipelineStageExecutionError,
    ResilienceProfileNotFoundError,
)
from src.core.explainability.explainability_models import (
    ExplanationMetadata,
    ExplanationResult,
    ExplanationSection,
)
from src.core.pipeline.pipeline_models import (
    PipelineExecutionContext,
    PipelineExecutionRequest,
    PipelineExecutionResult,
    PipelineRuntimeMetadata,
)
from src.core.pipeline.resilience import (
    FixedRetryStrategy,
    LogAndFailRecoveryStrategy,
    NullRecoveryStrategy,
    PipelineResilienceController,
    PipelineResilienceDefinition,
    PipelineResilienceProfile,
    PipelineResilienceProfileRegistry,
    RecoveryDefinition,
    ResilienceRuntimeMetadata,
    RetryDefinition,
    RetryExecutionTrace,
    ThreadPoolTimeoutPolicy,
    TimeoutDefinition,
)

# ==========================================
# Helper Mock Class/Functions
# ==========================================


class DummyOrchestrator:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def validate_compatibility(self, definition: Any) -> None:
        pass

    def execute(self, request: Any) -> PipelineExecutionResult:
        self.calls += 1
        outcome = (
            self.outcomes[self.calls - 1]
            if self.calls <= len(self.outcomes)
            else self.outcomes[-1]
        )
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, PipelineExecutionResult)
        return outcome


def create_dummy_execution_result(
    execution_id: str = "exec_1",
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
    ctx = PipelineExecutionContext(
        execution_id=execution_id,
        pipeline_id="pipe_1",
        claim="A claim",
        runtime_metadata=runtime_meta,
        stage_metadata=(),
        total_latency_ms=10.0,
        success=True,
    )
    return PipelineExecutionResult(
        evaluation_result=eval_result,
        execution_context=ctx,
    )


# ==========================================
# 1. Immutability Tests
# ==========================================


def test_models_immutability() -> None:
    retry_def = RetryDefinition(max_attempts=3, retry_delay_ms=10.0)
    with pytest.raises(ValidationError):
        setattr(retry_def, "max_attempts", 4)

    timeout_def = TimeoutDefinition(enabled=True, timeout_ms=500.0)
    with pytest.raises(ValidationError):
        setattr(timeout_def, "timeout_ms", 1000.0)

    trace = RetryExecutionTrace(
        execution_id="exec_1",
        total_attempts=1,
        succeeded=True,
        attempts=(),
        total_retry_overhead_ms=1.5,
    )
    with pytest.raises(ValidationError):
        setattr(trace, "succeeded", False)


# ==========================================
# 2. FixedRetryStrategy Tests
# ==========================================


def test_retry_strategy_success_first_attempt() -> None:
    dummy_res = create_dummy_execution_result()

    def fn() -> PipelineExecutionResult:
        return dummy_res

    strategy = FixedRetryStrategy()
    result, trace = strategy.execute_with_retry(
        fn, RetryDefinition(max_attempts=3, retry_delay_ms=10.0), "exec_1"
    )

    assert result is dummy_res
    assert trace.total_attempts == 1
    assert trace.succeeded is True
    assert len(trace.attempts) == 0


def test_retry_strategy_success_after_retries() -> None:
    dummy_res = create_dummy_execution_result()
    calls = []

    def fn() -> PipelineExecutionResult:
        calls.append(1)
        if len(calls) < 3:
            raise PipelineStageExecutionError("Transient error")
        return dummy_res

    sleep_calls = []

    def dummy_sleeper(seconds: float) -> None:
        sleep_calls.append(seconds)

    strategy = FixedRetryStrategy(
        retryable_types=(PipelineStageExecutionError,),
        sleeper=dummy_sleeper,
    )

    result, trace = strategy.execute_with_retry(
        fn, RetryDefinition(max_attempts=3, retry_delay_ms=50.0), "exec_1"
    )

    assert result is dummy_res
    assert len(calls) == 3
    assert trace.total_attempts == 3
    assert trace.succeeded is True
    assert len(trace.attempts) == 2
    assert trace.attempts[0].attempt_number == 1
    assert trace.attempts[1].attempt_number == 2
    assert sleep_calls == [0.05, 0.05]


def test_retry_strategy_exhausts_retries() -> None:
    def fn() -> PipelineExecutionResult:
        raise PipelineStageExecutionError("Permanent failure")

    def dummy_sleeper(seconds: float) -> None:
        pass

    strategy = FixedRetryStrategy(
        retryable_types=(PipelineStageExecutionError,),
        sleeper=dummy_sleeper,
    )

    with pytest.raises(PipelineStageExecutionError) as exc_info:
        strategy.execute_with_retry(
            fn, RetryDefinition(max_attempts=3, retry_delay_ms=10.0), "exec_1"
        )

    # Verify trace was attached to the raised exception
    assert hasattr(exc_info.value, "retry_trace")
    trace = exc_info.value.retry_trace
    assert trace.total_attempts == 3
    assert trace.succeeded is False
    assert len(trace.attempts) == 3


def test_retry_strategy_propagates_non_retryable_immediately() -> None:
    calls = 0

    def fn() -> PipelineExecutionResult:
        nonlocal calls
        calls += 1
        raise ValueError("Non-retryable domain error")

    strategy = FixedRetryStrategy(
        retryable_types=(PipelineStageExecutionError,),
    )

    with pytest.raises(ValueError):
        strategy.execute_with_retry(
            fn, RetryDefinition(max_attempts=3, retry_delay_ms=10.0), "exec_1"
        )

    assert calls == 1


# ==========================================
# 3. ThreadPoolTimeoutPolicy Tests
# ==========================================


def test_timeout_policy_completes_successfully() -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    policy = ThreadPoolTimeoutPolicy(executor)
    dummy_res = create_dummy_execution_result()

    def fn() -> PipelineExecutionResult:
        return dummy_res

    result = policy.execute_with_timeout(
        fn, TimeoutDefinition(enabled=True, timeout_ms=500.0)
    )
    assert result is dummy_res
    executor.shutdown()


def test_timeout_policy_raises_timeout_error() -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    policy = ThreadPoolTimeoutPolicy(executor)

    def slow_fn() -> PipelineExecutionResult:
        time.sleep(0.2)
        return create_dummy_execution_result()

    with pytest.raises(PipelineResilienceTimeoutError):
        policy.execute_with_timeout(
            slow_fn, TimeoutDefinition(enabled=True, timeout_ms=50.0)
        )
    executor.shutdown()


def test_timeout_policy_disabled() -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    policy = ThreadPoolTimeoutPolicy(executor)
    dummy_res = create_dummy_execution_result()

    def fn() -> PipelineExecutionResult:
        return dummy_res

    # Should run directly and succeed immediately
    result = policy.execute_with_timeout(
        fn, TimeoutDefinition(enabled=False, timeout_ms=50.0)
    )
    assert result is dummy_res
    executor.shutdown()


# ==========================================
# 4. Recovery Strategy Tests
# ==========================================


def test_null_recovery_strategy() -> None:
    strategy = NullRecoveryStrategy()
    strategy.validate_compatibility(RecoveryDefinition(strategy_id="default_recovery"))

    trace = RetryExecutionTrace(
        execution_id="exec_1",
        total_attempts=3,
        succeeded=False,
        attempts=(),
        total_retry_overhead_ms=12.0,
        terminal_error="PipelineStageExecutionError: Failed",
    )

    req = PipelineExecutionRequest(claim="Test", pipeline_profile_id="default")
    res = strategy.recover(
        req,
        trace,
        RecoveryDefinition(strategy_id="default_recovery"),
        "pipe_1",
        "res_1",
        True,
    )

    assert res.succeeded is False
    assert res.recovery_strategy_id == "default_recovery"
    assert res.failure_reason == "PipelineStageExecutionError: Failed"
    assert res.resilience_metadata.pipeline_profile_id == "pipe_1"
    assert res.resilience_metadata.resilience_profile_id == "res_1"
    assert res.resilience_metadata.timeout_enforced is True


def test_log_and_fail_recovery_strategy(caplog: pytest.LogCaptureFixture) -> None:
    strategy = LogAndFailRecoveryStrategy()
    strategy.validate_compatibility(
        RecoveryDefinition(strategy_id="log_and_fail_recovery")
    )

    trace = RetryExecutionTrace(
        execution_id="exec_1",
        total_attempts=3,
        succeeded=False,
        attempts=(),
        total_retry_overhead_ms=10.0,
        terminal_error="PipelineStageExecutionError: Bad stage",
    )

    req = PipelineExecutionRequest(claim="Test", pipeline_profile_id="default")
    with caplog.at_level(logging.ERROR, logger="arbiter.resilience"):
        res = strategy.recover(
            req,
            trace,
            RecoveryDefinition(strategy_id="log_and_fail_recovery"),
            "pipe_1",
            "res_1",
            True,
        )

    assert res.succeeded is False
    assert len(caplog.records) == 1
    assert "Resilience recovery invoked for execution exec_1" in caplog.text


# ==========================================
# 5. Registry & Profile Tests
# ==========================================


def test_resilience_profile_and_registry() -> None:
    retry_def = RetryDefinition()
    timeout_def = TimeoutDefinition()
    recovery_def = RecoveryDefinition(strategy_id="default_recovery")
    resilience_def = PipelineResilienceDefinition(
        enabled=True, retry=retry_def, timeout=timeout_def, recovery=recovery_def
    )
    strategy = NullRecoveryStrategy()

    profile = PipelineResilienceProfile(
        profile_id="res_profile_1",
        definition=resilience_def,
        recovery_strategy=strategy,
    )

    registry = PipelineResilienceProfileRegistry(profiles=(profile,))
    assert registry.resolve("res_profile_1") is profile

    with pytest.raises(DuplicateResilienceProfileError):
        PipelineResilienceProfileRegistry(profiles=(profile, profile))

    with pytest.raises(ResilienceProfileNotFoundError):
        registry.resolve("unknown")


# ==========================================
# 6. PipelineResilienceController Tests
# ==========================================


def test_controller_success_path() -> None:
    # Set up mock strategies
    dummy_res = create_dummy_execution_result("exec_1")
    dummy_orchestrator = DummyOrchestrator([dummy_res])

    def dummy_sleeper(seconds: float) -> None:
        pass

    retry_strategy = FixedRetryStrategy(sleeper=dummy_sleeper)
    executor = ThreadPoolExecutor(max_workers=1)
    timeout_policy = ThreadPoolTimeoutPolicy(executor)

    controller = PipelineResilienceController(retry_strategy, timeout_policy)

    retry_def = RetryDefinition(max_attempts=3, retry_delay_ms=10.0)
    timeout_def = TimeoutDefinition(enabled=True, timeout_ms=500.0)
    recovery_def = RecoveryDefinition(strategy_id="default_recovery")
    resilience_def = PipelineResilienceDefinition(
        enabled=True, retry=retry_def, timeout=timeout_def, recovery=recovery_def
    )

    profile = PipelineResilienceProfile(
        profile_id="default_resilience",
        definition=resilience_def,
        recovery_strategy=NullRecoveryStrategy(),
    )

    req = PipelineExecutionRequest(claim="Test", pipeline_profile_id="default_pipeline")
    result = controller.execute(req, dummy_orchestrator, profile)

    assert result is dummy_res
    assert hasattr(result, "resilience_metadata")
    meta = getattr(result, "resilience_metadata")
    assert isinstance(meta, ResilienceRuntimeMetadata)
    assert meta.retry_trace.succeeded is True
    assert meta.retry_trace.total_attempts == 1
    assert meta.recovery_invoked is False

    executor.shutdown()


def test_controller_fallback_recovery_path() -> None:
    # Set up mock orchestrator that always fails
    dummy_orchestrator = DummyOrchestrator(
        [PipelineStageExecutionError("Failed stage")]
    )

    def dummy_sleeper(seconds: float) -> None:
        pass

    retry_strategy = FixedRetryStrategy(sleeper=dummy_sleeper)
    executor = ThreadPoolExecutor(max_workers=1)
    timeout_policy = ThreadPoolTimeoutPolicy(executor)

    controller = PipelineResilienceController(retry_strategy, timeout_policy)

    retry_def = RetryDefinition(max_attempts=2, retry_delay_ms=5.0)
    timeout_def = TimeoutDefinition(enabled=True, timeout_ms=100.0)
    recovery_def = RecoveryDefinition(strategy_id="default_recovery")
    resilience_def = PipelineResilienceDefinition(
        enabled=True, retry=retry_def, timeout=timeout_def, recovery=recovery_def
    )

    profile = PipelineResilienceProfile(
        profile_id="default_resilience",
        definition=resilience_def,
        recovery_strategy=NullRecoveryStrategy(),
    )

    req = PipelineExecutionRequest(claim="Test", pipeline_profile_id="default_pipeline")

    with pytest.raises(PipelineStageExecutionError) as exc_info:
        controller.execute(req, dummy_orchestrator, profile)

    assert "Recovery strategy 'default_recovery' failed" in str(exc_info.value)
    assert hasattr(exc_info.value, "resilience_metadata")
    meta = getattr(exc_info.value, "resilience_metadata")
    assert isinstance(meta, ResilienceRuntimeMetadata)
    assert meta.recovery_invoked is True
    assert meta.recovery_strategy_id == "default_recovery"
    assert meta.retry_trace.succeeded is False
    assert meta.retry_trace.total_attempts == 2

    executor.shutdown()


# ==========================================
# 7. Bootstrap & Lifespan Integration Tests
# ==========================================


def test_bootstrap_integration() -> None:
    settings = Settings()
    executor = ThreadPoolExecutor(max_workers=1)

    # 1. Build resilience registry
    registry = build_resilience_registry(settings, executor)
    assert isinstance(registry, PipelineResilienceProfileRegistry)

    # 2. Build resilience controller
    controller = build_resilience_controller(settings, executor)
    assert isinstance(controller, PipelineResilienceController)

    executor.shutdown()


def test_api_integration_e2e() -> None:
    app = create_app()
    with TestClient(app) as client:
        assert hasattr(app.state, "resilience_registry")
        assert hasattr(app.state, "resilience_controller")

        # Trigger an E2E evaluate call
        payload = {
            "claim": "FastAPI with retry controls.",
            "pipeline_profile_id": "default_pipeline",
        }
        response = client.post("/v1/evaluate", json=payload)
        assert response.status_code == 200
