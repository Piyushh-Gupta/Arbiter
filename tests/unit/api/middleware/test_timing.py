"""Tests for TimingComponent."""

import pytest

from src.api.middleware.base import Clock
from src.api.middleware.middleware_models import (
    MiddlewareExecutionContext,
    RequestLifecyclePhase,
)
from src.api.middleware.timing import TimingComponent
from src.core.exceptions import InvalidLifecycleTransitionError


class MockClock(Clock):
    def __init__(self, time_ns: int = 1000) -> None:
        self.time_ns = time_ns

    def now_ns(self) -> int:
        return self.time_ns


def test_timing_component_records_start_time() -> None:
    clock = MockClock(time_ns=5000)
    component = TimingComponent(clock=clock)
    context = MiddlewareExecutionContext(
        request={},
        phase=RequestLifecyclePhase.REQUEST_RECEIVED,
    )

    new_context = component.execute(context)
    assert new_context.timing.start_time_ns == 5000


def test_timing_component_invalid_phase() -> None:
    clock = MockClock()
    component = TimingComponent(clock=clock)
    context = MiddlewareExecutionContext(
        request={},
        phase=RequestLifecyclePhase.FINALIZED,
    )

    with pytest.raises(InvalidLifecycleTransitionError):
        component.execute(context)
