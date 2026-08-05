import pytest
from pydantic import ValidationError

from src.core.exceptions import (
    DuplicateOperationalProfileError,
    OperationalProfileNotFoundError,
    PipelineOperationalConfigurationError,
)
from src.core.pipeline.operations.operation_models import PipelineOperationalDefinition
from src.core.pipeline.operations.profiles import (
    PipelineOperationalProfile,
    PipelineOperationalProfileRegistry,
)


def test_pipeline_operational_profile_validation() -> None:
    # Valid profile
    profile = PipelineOperationalProfile(
        profile_id="valid",
        definition=PipelineOperationalDefinition(
            startup_timeout_seconds=10.0,
            shutdown_timeout_seconds=10.0,
            health_check_timeout_seconds=5.0,
            require_all_subsystems_ready=True,
        ),
    )
    assert profile.profile_id == "valid"

    # Missing ID
    with pytest.raises(
        PipelineOperationalConfigurationError, match="profile_id cannot be empty"
    ):
        PipelineOperationalProfile(
            profile_id="", definition=PipelineOperationalDefinition()
        )

    # Invalid startup timeout
    with pytest.raises(ValidationError):
        PipelineOperationalProfile(
            profile_id="invalid",
            definition=PipelineOperationalDefinition(startup_timeout_seconds=-1.0),
        )

    # Invalid shutdown timeout
    with pytest.raises(ValidationError):
        PipelineOperationalProfile(
            profile_id="invalid",
            definition=PipelineOperationalDefinition(shutdown_timeout_seconds=-1.0),
        )

    # Invalid health check timeout
    with pytest.raises(ValidationError):
        PipelineOperationalProfile(
            profile_id="invalid",
            definition=PipelineOperationalDefinition(
                health_check_timeout_seconds=-1.0
            ),
        )


def test_pipeline_operational_profile_registry() -> None:
    profile1 = PipelineOperationalProfile(
        profile_id="p1", definition=PipelineOperationalDefinition()
    )
    profile2 = PipelineOperationalProfile(
        profile_id="p2", definition=PipelineOperationalDefinition()
    )

    registry = PipelineOperationalProfileRegistry((profile1,))
    assert len(registry.profiles) == 1

    registry.register(profile2)
    assert len(registry.profiles) == 2

    assert registry.resolve("p1") == profile1
    assert registry.resolve("p2") == profile2

    with pytest.raises(DuplicateOperationalProfileError):
        registry.register(profile1)

    with pytest.raises(OperationalProfileNotFoundError):
        registry.resolve("nonexistent")
