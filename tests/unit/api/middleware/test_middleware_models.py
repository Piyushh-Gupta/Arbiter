"""Tests for middleware models."""

import pytest
from pydantic import ValidationError

from src.api.middleware.middleware_models import (
    CorrelationContext,
    MiddlewareExecutionContext,
    MiddlewareProfile,
    RequestLifecyclePhase,
    RequestTiming,
)


def test_correlation_context_immutability() -> None:
    context = CorrelationContext(correlation_id="test-123")
    assert context.correlation_id == "test-123"
    assert context.client_id is None

    with pytest.raises(ValidationError):
        context.correlation_id = "new-id"


def test_request_timing() -> None:
    timing = RequestTiming(start_time_ns=1000000, end_time_ns=2500000)
    assert timing.start_time_ns == 1000000
    assert timing.end_time_ns == 2500000
    assert timing.elapsed_ms == 1.5


def test_request_timing_zero_elapsed() -> None:
    timing = RequestTiming()
    assert timing.elapsed_ms == 0.0


def test_middleware_execution_context() -> None:
    request_obj: dict[str, dict[str, str]] = {"headers": {}}
    context = MiddlewareExecutionContext(
        request=request_obj,
    )
    assert context.request == request_obj
    assert context.phase == RequestLifecyclePhase.REQUEST_RECEIVED
    assert context.correlation_context is None
    assert context.contract_profile_id is None

    with pytest.raises(ValidationError):
        context.phase = RequestLifecyclePhase.FINALIZED


def test_middleware_profile() -> None:
    profile = MiddlewareProfile(profile_id="test-profile")
    assert profile.profile_id == "test-profile"
    assert profile.require_correlation_propagation is True
