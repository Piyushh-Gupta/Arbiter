"""Tests for CorrelationComponent."""

import pytest

from src.api.middleware.correlation import CorrelationComponent
from src.api.middleware.middleware_models import (
    MiddlewareExecutionContext,
    RequestLifecyclePhase,
)
from src.core.exceptions import InvalidLifecycleTransitionError


def test_correlation_component_extracts_headers() -> None:
    component = CorrelationComponent()
    request: dict[str, dict[str, str]] = {
        "headers": {
            "X-Correlation-ID": "test-corr-id",
            "X-Client-ID": "test-client",
        }
    }
    context = MiddlewareExecutionContext(
        request=request,
        phase=RequestLifecyclePhase.REQUEST_RECEIVED,
    )

    new_context = component.execute(context)

    assert new_context.phase == RequestLifecyclePhase.CORRELATION_ESTABLISHED
    assert new_context.correlation_context is not None
    assert new_context.correlation_context.correlation_id == "test-corr-id"
    assert new_context.correlation_context.client_id == "test-client"


def test_correlation_component_generates_id() -> None:
    component = CorrelationComponent()
    request: dict[str, dict[str, str]] = {"headers": {}}
    context = MiddlewareExecutionContext(
        request=request,
        phase=RequestLifecyclePhase.REQUEST_RECEIVED,
    )

    new_context = component.execute(context)

    assert new_context.phase == RequestLifecyclePhase.CORRELATION_ESTABLISHED
    assert new_context.correlation_context is not None
    assert new_context.correlation_context.correlation_id is not None
    assert len(new_context.correlation_context.correlation_id) > 0


def test_correlation_component_invalid_phase() -> None:
    component = CorrelationComponent()
    context = MiddlewareExecutionContext(
        request={},
        phase=RequestLifecyclePhase.FINALIZED,
    )

    with pytest.raises(InvalidLifecycleTransitionError):
        component.execute(context)
