"""Profile and definition models for telemetry exporters."""

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from src.core.exceptions import (
    DuplicateTelemetryProfileError,
    TelemetryProfileNotFoundError,
)
from src.core.pipeline.telemetry.base import BaseTelemetryExporter


class TelemetryExporterDefinition(BaseModel):
    """Base immutable configuration for a telemetry exporter."""

    exporter_id: str = Field(..., min_length=1)
    enabled: bool = Field(default=True)
    model_config = ConfigDict(frozen=True)


class LogTelemetryExporterDefinition(TelemetryExporterDefinition):
    """Configuration for structured log-based telemetry export."""

    log_level: str = Field(default="INFO")
    include_stage_breakdown: bool = Field(default=True)
    model_config = ConfigDict(frozen=True)


class JsonTelemetryExporterDefinition(TelemetryExporterDefinition):
    """Configuration for JSON file telemetry export."""

    output_path: str = Field(..., min_length=1)
    pretty_print: bool = Field(default=False)
    model_config = ConfigDict(frozen=True)


class TelemetryExporterProfile(BaseModel):
    """Immutable binding of a TelemetryExporterDefinition to a concrete exporter."""

    profile_id: str = Field(..., min_length=1)
    definition: TelemetryExporterDefinition = Field(...)
    exporter: BaseTelemetryExporter = Field(...)
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "TelemetryExporterProfile":
        """Front-loads compatibility validation at profile construction."""
        self.exporter.validate_compatibility(self.definition)
        return self


class TelemetryExporterRegistry(BaseModel):
    """Registry providing O(1) exporter profile resolution."""

    profiles: tuple[TelemetryExporterProfile, ...] = Field(..., min_length=1)
    _profile_index: dict[str, TelemetryExporterProfile] = PrivateAttr(
        default_factory=dict
    )
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_index(self) -> "TelemetryExporterRegistry":
        """Detects duplicate profile_ids at construction."""
        index = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicateTelemetryProfileError(
                    f"Duplicate telemetry profile ID: {profile.profile_id}"
                )
            index[profile.profile_id] = profile
        self._profile_index = index
        return self

    def resolve(self, profile_id: str) -> TelemetryExporterProfile:
        """Resolves an exporter profile.

        Raises:
            TelemetryProfileNotFoundError: If profile_id is absent.
        """
        if profile_id not in self._profile_index:
            raise TelemetryProfileNotFoundError(
                f"Telemetry profile '{profile_id}' not found in registry."
            )
        return self._profile_index[profile_id]


class PipelineTelemetryDefinition(BaseModel):
    """Top-level immutable telemetry configuration governing collector and exporter wiring."""

    enabled: bool = Field(default=True)
    snapshot_on_every_execution: bool = Field(default=False)
    active_exporter_profile_ids: tuple[str, ...] = Field(default_factory=tuple)
    model_config = ConfigDict(frozen=True)
