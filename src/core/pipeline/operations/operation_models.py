from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PipelineLifecycleState(str, Enum):
    """Deterministic state-machine states for the pipeline lifecycle."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    FAILED = "failed"


class PipelineHealthStatus(str, Enum):
    """Overall pipeline or subsystem health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class PipelineReadinessStatus(str, Enum):
    """Readiness status indicating if the pipeline can accept traffic."""

    READY = "ready"
    NOT_READY = "not_ready"


class PipelineOperationalMetadata(BaseModel):
    """Immutable operational metadata for the pipeline environment."""

    model_config = ConfigDict(frozen=True)

    environment: str = Field(default="production")
    version: str = Field(default="unknown")
    host_id: str | None = Field(default=None)
    tags: dict[str, str] = Field(default_factory=dict)


class SubsystemHealthRecord(BaseModel):
    """Immutable health record for a specific subsystem."""

    model_config = ConfigDict(frozen=True)

    subsystem_id: str
    health_status: PipelineHealthStatus
    readiness_status: PipelineReadinessStatus
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class PipelineOperationalSnapshot(BaseModel):
    """Immutable snapshot of the pipeline's operational state."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    lifecycle_state: PipelineLifecycleState
    overall_health: PipelineHealthStatus
    overall_readiness: PipelineReadinessStatus
    subsystem_records: tuple[SubsystemHealthRecord, ...] = Field(default_factory=tuple)
    metadata: PipelineOperationalMetadata


class PipelineOperationalDefinition(BaseModel):
    """Defines operational constraints for the pipeline."""

    model_config = ConfigDict(frozen=True)

    startup_timeout_seconds: float = Field(default=30.0, ge=0.0)
    shutdown_timeout_seconds: float = Field(default=30.0, ge=0.0)
    health_check_timeout_seconds: float = Field(default=5.0, ge=0.0)
    require_all_subsystems_ready: bool = Field(default=True)
