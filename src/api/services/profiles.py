"""Service layer configuration profiles."""

from pydantic import BaseModel, ConfigDict


class ServiceProfile(BaseModel):
    """Immutable service configuration profile."""

    model_config = ConfigDict(frozen=True)
    profile_id: str
    require_correlation_id: bool = True
    timeout_seconds: float = 30.0
