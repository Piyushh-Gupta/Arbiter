"""Unit tests for the API Contract versioning registry."""

import pytest

from src.api.contracts.versioning import (
    ApiContractDefinition,
    ApiContractProfile,
    ApiContractRegistry,
)
from src.core.exceptions import (
    ApiContractProfileNotFoundError,
    DuplicateApiContractProfileError,
)


def test_registry_initialization() -> None:
    """Verifies successful registry initialization."""
    definition = ApiContractDefinition()
    profile = ApiContractProfile(profile_id="test", definition=definition)
    registry = ApiContractRegistry([profile])

    resolved = registry.resolve("test")
    assert resolved == profile


def test_registry_duplicate_profiles() -> None:
    """Verifies registry rejects duplicate profile IDs."""
    definition = ApiContractDefinition()
    profile1 = ApiContractProfile(profile_id="test", definition=definition)
    profile2 = ApiContractProfile(profile_id="test", definition=definition)

    with pytest.raises(DuplicateApiContractProfileError):
        ApiContractRegistry([profile1, profile2])


def test_registry_not_found() -> None:
    """Verifies registry raises error for unknown profiles."""
    registry = ApiContractRegistry([])

    with pytest.raises(ApiContractProfileNotFoundError):
        registry.resolve("unknown")
