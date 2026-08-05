"""Tests for middleware registry and pipeline."""

import pytest

from src.api.middleware.base import BaseMiddlewareComponent
from src.api.middleware.middleware_models import (
    MiddlewareExecutionContext,
    MiddlewareProfile,
    RequestLifecyclePhase,
)
from src.api.middleware.pipeline import MiddlewarePipeline
from src.api.middleware.registry import MiddlewareProfileRegistry
from src.core.exceptions import (
    DuplicateMiddlewareProfileError,
    MiddlewareConfigurationError,
    MiddlewareProfileNotFoundError,
)


class DummyComponent(BaseMiddlewareComponent):
    def execute(
        self, context: MiddlewareExecutionContext
    ) -> MiddlewareExecutionContext:
        return context.model_copy(update={"phase": RequestLifecyclePhase.VALIDATED})


def test_registry_initialization_and_resolution() -> None:
    profile1 = MiddlewareProfile(profile_id="p1")
    profile2 = MiddlewareProfile(profile_id="p2")
    registry = MiddlewareProfileRegistry(profiles=[profile1, profile2])

    resolved = registry.resolve("p1")
    assert resolved == profile1

    with pytest.raises(MiddlewareProfileNotFoundError):
        registry.resolve("non-existent")


def test_registry_duplicate_profiles() -> None:
    profile1 = MiddlewareProfile(profile_id="p1")
    profile2 = MiddlewareProfile(profile_id="p1")

    with pytest.raises(DuplicateMiddlewareProfileError):
        MiddlewareProfileRegistry(profiles=[profile1, profile2])


def test_pipeline_execution() -> None:
    component = DummyComponent()
    pipeline = MiddlewarePipeline(components=[component])

    context = MiddlewareExecutionContext(
        request={},
        phase=RequestLifecyclePhase.REQUEST_RECEIVED,
    )

    final_context = pipeline.execute(context)
    assert final_context.phase == RequestLifecyclePhase.VALIDATED


def test_pipeline_empty_components() -> None:
    with pytest.raises(MiddlewareConfigurationError):
        MiddlewarePipeline(components=[])
