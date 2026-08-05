"""Tests for LifecycleManager."""

import pytest

from src.api.middleware.base import BaseMiddlewareComponent, Clock
from src.api.middleware.lifecycle import LifecycleManager
from src.api.middleware.middleware_models import RequestLifecyclePhase
from src.api.middleware.pipeline import MiddlewarePipeline
from src.core.exceptions import InvalidLifecycleTransitionError


class MockClock(Clock):
    def __init__(self, time_ns: int = 1000) -> None:
        self.time_ns = time_ns

    def now_ns(self) -> int:
        return self.time_ns


def test_lifecycle_manager_initializes_and_finalizes_request() -> None:
    from src.api.middleware.middleware_models import MiddlewareExecutionContext

    class Dummy(BaseMiddlewareComponent):
        def execute(
            self, ctx: MiddlewareExecutionContext
        ) -> MiddlewareExecutionContext:
            return ctx

    pipeline = MiddlewarePipeline(components=[Dummy()])
    clock = MockClock(time_ns=1000)
    manager = LifecycleManager(pipeline=pipeline, clock=clock, active_profile_id="test")

    context = manager.initialize_request(request={})
    assert context.phase == RequestLifecyclePhase.REQUEST_RECEIVED
    assert context.contract_profile_id == "test"

    clock.time_ns = 5000
    final_context = manager.finalize_request(context)

    assert final_context.phase == RequestLifecyclePhase.FINALIZED
    assert final_context.timing.end_time_ns == 5000


def test_lifecycle_manager_cannot_finalize_twice() -> None:
    from src.api.middleware.middleware_models import MiddlewareExecutionContext

    class Dummy(BaseMiddlewareComponent):
        def execute(
            self, ctx: MiddlewareExecutionContext
        ) -> MiddlewareExecutionContext:
            return ctx

    pipeline = MiddlewarePipeline(components=[Dummy()])
    clock = MockClock(time_ns=1000)
    manager = LifecycleManager(pipeline=pipeline, clock=clock, active_profile_id="test")

    context = manager.initialize_request(request={})
    final_context = manager.finalize_request(context)

    with pytest.raises(InvalidLifecycleTransitionError):
        manager.finalize_request(final_context)
