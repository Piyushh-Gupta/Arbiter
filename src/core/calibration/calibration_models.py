"""Immutable domain models for the Calibration subsystem."""

import math
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class CalibrationStrategyType(str, Enum):
    """Supported confidence calibration strategies."""

    IDENTITY = "IDENTITY"
    TEMPERATURE_SCALING = "TEMPERATURE_SCALING"
    PLATT_SCALING = "PLATT_SCALING"
    ISOTONIC_CALIBRATION = "ISOTONIC_CALIBRATION"


class TemperatureScalingParameters(BaseModel):
    """Parameters for Temperature Scaling calibration."""

    temperature: float = Field(..., description="Scaling temperature parameter.")

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_parameters(self) -> "TemperatureScalingParameters":
        if math.isnan(self.temperature) or math.isinf(self.temperature):
            raise ValueError("Temperature must be finite.")
        if self.temperature <= 0.0:
            raise ValueError("Temperature must be strictly positive (> 0).")
        return self


class PlattScalingParameters(BaseModel):
    """Parameters for Platt Scaling (sigmoid) calibration."""

    slope: float = Field(..., description="Sigmoid slope (scaling) parameter.")
    intercept: float = Field(..., description="Sigmoid intercept parameter.")

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_parameters(self) -> "PlattScalingParameters":
        for name, val in [("slope", self.slope), ("intercept", self.intercept)]:
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"{name} must be finite.")
        return self


class IsotonicCalibrationParameters(BaseModel):
    """Parameters for Isotonic (monotonic piecewise constant) calibration."""

    x_thresholds: tuple[float, ...] = Field(
        ..., description="Ascending uncalibrated threshold values."
    )
    y_values: tuple[float, ...] = Field(
        ..., description="Corresponding calibrated target probabilities."
    )

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_parameters(self) -> "IsotonicCalibrationParameters":
        if len(self.x_thresholds) != len(self.y_values):
            raise ValueError("x_thresholds and y_values must have identical lengths.")
        if not self.x_thresholds:
            raise ValueError("Isotonic mappings cannot be empty.")

        # Validate finiteness
        for val in self.x_thresholds:
            if math.isnan(val) or math.isinf(val):
                raise ValueError("x_thresholds values must be finite.")
        for val in self.y_values:
            if math.isnan(val) or math.isinf(val):
                raise ValueError("y_values must be finite.")

        # Verify strictly increasing x
        for i in range(len(self.x_thresholds) - 1):
            if self.x_thresholds[i] >= self.x_thresholds[i + 1]:
                raise ValueError("x_thresholds must be strictly increasing.")

        # Verify monotonic non-decreasing y
        for i in range(len(self.y_values) - 1):
            if self.y_values[i] > self.y_values[i + 1]:
                raise ValueError("y_values must be monotonically non-decreasing.")

        return self


class CalibrationDefinition(BaseModel):
    """Configuration mapping calibration strategy to active parameters."""

    strategy: CalibrationStrategyType = Field(
        ..., description="Target calibration strategy type."
    )
    calibration_version: str = Field(
        default="1.0", description="Semantic calibration version."
    )
    parameters: Any = Field(
        default=None, description="Typed parameters model appropriate for the strategy."
    )
    confidence_bounds: tuple[float, float] = Field(
        default=(0.0, 1.0), description="Clamping boundaries for confidence."
    )
    uncertainty_method: str = Field(
        default="CONFIDENCE_MARGIN", description="Name of uncertainty estimator."
    )

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_definition(self) -> "CalibrationDefinition":
        # Validate that parameters match the strategy
        if self.strategy == CalibrationStrategyType.TEMPERATURE_SCALING:
            if not isinstance(self.parameters, TemperatureScalingParameters):
                raise ValueError("Parameters must be TemperatureScalingParameters.")
        elif self.strategy == CalibrationStrategyType.PLATT_SCALING:
            if not isinstance(self.parameters, PlattScalingParameters):
                raise ValueError("Parameters must be PlattScalingParameters.")
        elif self.strategy == CalibrationStrategyType.ISOTONIC_CALIBRATION:
            if not isinstance(self.parameters, IsotonicCalibrationParameters):
                raise ValueError("Parameters must be IsotonicCalibrationParameters.")
        return self


class CalibrationTrace(BaseModel):
    """Immutable sequence record of confidence conversion steps."""

    original_confidence: float | None = Field(
        ..., description="Confidence prior to calibration."
    )
    intermediate_values: dict[str, float] = Field(
        ..., description="Dictionary of intermediate values computed."
    )
    final_confidence: float = Field(
        ..., description="Final calibrated confidence score."
    )
    applied_strategy: CalibrationStrategyType = Field(
        ..., description="Strategy type applied."
    )
    parameter_version: str = Field(..., description="Active version tag of parameters.")

    model_config = ConfigDict(frozen=True)


class CalibrationResult(BaseModel):
    """Final calibrated result container containing transformed values and execution trace."""

    original_confidence: float | None = Field(..., description="Raw confidence value.")
    calibrated_confidence: float = Field(
        ..., description="Calibrated confidence value in [0, 1]."
    )
    uncertainty_estimate: float = Field(
        ..., description="Computed uncertainty estimate in [0, 1]."
    )
    calibration_trace: CalibrationTrace = Field(
        ..., description="Detailed transformation trace."
    )
    calibration_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata dictionary."
    )

    model_config = ConfigDict(frozen=True)


class CalibrationProfile(BaseModel):
    """Immutable binder pairing a profile name with its configuration and execution strategies."""

    profile_id: str = Field(..., description="Unique profile identifier.")
    definition: CalibrationDefinition = Field(
        ..., description="Calibration configuration definition."
    )
    strategy: Any = Field(
        ..., description="The executable strategy strategy (BaseCalibrationStrategy)."
    )
    uncertainty_estimator: Any = Field(
        ...,
        description="The executable uncertainty estimator (BaseUncertaintyEstimator).",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_profile(self) -> "CalibrationProfile":
        # Check compatible strategies and estimators
        from src.core.calibration.base import (
            BaseCalibrationStrategy,
            BaseUncertaintyEstimator,
        )

        if not isinstance(self.strategy, BaseCalibrationStrategy):
            raise ValueError(
                "strategy must implement BaseCalibrationStrategy protocol."
            )
        if not isinstance(self.uncertainty_estimator, BaseUncertaintyEstimator):
            raise ValueError(
                "uncertainty_estimator must implement BaseUncertaintyEstimator protocol."
            )
        return self


class CalibrationProfileRegistry(BaseModel):
    """Immutable O(1) profile registry for named calibration pipelines."""

    profiles: tuple[CalibrationProfile, ...] = Field(
        ..., min_length=1, description="Registered profiles."
    )

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_unique_profiles(self) -> "CalibrationProfileRegistry":
        seen = set()
        for p in self.profiles:
            if p.profile_id in seen:
                raise ValueError(f"Duplicate profile_id '{p.profile_id}' in registry.")
            seen.add(p.profile_id)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def _by_id(self) -> dict[str, CalibrationProfile]:
        return {p.profile_id: p for p in self.profiles}

    def resolve(self, profile_id: str) -> CalibrationProfile:
        if profile_id not in self._by_id:
            raise KeyError(f"Calibration profile '{profile_id}' not found in registry.")
        return self._by_id[profile_id]
