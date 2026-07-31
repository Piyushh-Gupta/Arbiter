"""Confidence Calibration & Uncertainty Estimation subsystem."""

from src.core.calibration.base import BaseCalibrationStrategy, BaseUncertaintyEstimator
from src.core.calibration.calibration_models import (
    CalibrationDefinition,
    CalibrationProfile,
    CalibrationProfileRegistry,
    CalibrationResult,
    CalibrationStrategyType,
    CalibrationTrace,
    IsotonicCalibrationParameters,
    PlattScalingParameters,
    TemperatureScalingParameters,
)
from src.core.calibration.implementations import (
    ConfidenceMarginEstimator,
    EntropyEstimator,
    IdentityCalibration,
    IsotonicCalibration,
    NormalizedVarianceEstimator,
    PlattScalingCalibration,
    TemperatureScalingCalibration,
)

__all__ = [
    "BaseCalibrationStrategy",
    "BaseUncertaintyEstimator",
    "CalibrationDefinition",
    "CalibrationProfile",
    "CalibrationProfileRegistry",
    "CalibrationResult",
    "CalibrationStrategyType",
    "CalibrationTrace",
    "IsotonicCalibrationParameters",
    "PlattScalingParameters",
    "TemperatureScalingParameters",
    "ConfidenceMarginEstimator",
    "EntropyEstimator",
    "IdentityCalibration",
    "IsotonicCalibration",
    "NormalizedVarianceEstimator",
    "PlattScalingCalibration",
    "TemperatureScalingCalibration",
]
