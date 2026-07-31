"""Stateless concrete implementations of calibration strategies and uncertainty estimators."""

import bisect
import math

from src.core.calibration.base import BaseCalibrationStrategy, BaseUncertaintyEstimator
from src.core.calibration.calibration_models import (
    CalibrationDefinition,
    CalibrationResult,
    CalibrationStrategyType,
    CalibrationTrace,
    IsotonicCalibrationParameters,
    PlattScalingParameters,
    TemperatureScalingParameters,
)
from src.core.verification.verification_models import VerificationResult


class EntropyEstimator(BaseUncertaintyEstimator):
    """Computes binary Shannon entropy normalized to [0, 1]."""

    def estimate_uncertainty(self, calibrated_confidence: float) -> float:
        p = max(1e-15, min(1.0 - 1e-15, calibrated_confidence))
        val = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
        return float(val)


class ConfidenceMarginEstimator(BaseUncertaintyEstimator):
    """Computes linear uncertainty margin (1.0 - confidence)."""

    def estimate_uncertainty(self, calibrated_confidence: float) -> float:
        return float(1.0 - calibrated_confidence)


class NormalizedVarianceEstimator(BaseUncertaintyEstimator):
    """Computes normalized variance (4p(1-p))."""

    def estimate_uncertainty(self, calibrated_confidence: float) -> float:
        return float(4.0 * calibrated_confidence * (1.0 - calibrated_confidence))


class IdentityCalibration(BaseCalibrationStrategy):
    """Identity calibration strategy returning the original confidence unchanged."""

    def __init__(
        self, uncertainty_estimator: BaseUncertaintyEstimator | None = None
    ) -> None:
        self.uncertainty_estimator = (
            uncertainty_estimator or ConfidenceMarginEstimator()
        )

    def calibrate(
        self,
        result: VerificationResult,
        definition: CalibrationDefinition,
        uncertainty_estimator: BaseUncertaintyEstimator | None = None,
    ) -> CalibrationResult:
        p = result.confidence if result.confidence is not None else 0.0
        estimator = uncertainty_estimator or self.uncertainty_estimator
        u = estimator.estimate_uncertainty(p)
        trace = CalibrationTrace(
            original_confidence=result.confidence,
            intermediate_values={},
            final_confidence=p,
            applied_strategy=CalibrationStrategyType.IDENTITY,
            parameter_version=definition.calibration_version,
        )
        return CalibrationResult(
            original_confidence=result.confidence,
            calibrated_confidence=p,
            uncertainty_estimate=u,
            calibration_trace=trace,
            calibration_metadata={},
        )


class TemperatureScalingCalibration(BaseCalibrationStrategy):
    """Temperature scaling calibration strategy applying sigmoid(logit / T)."""

    def __init__(
        self, uncertainty_estimator: BaseUncertaintyEstimator | None = None
    ) -> None:
        self.uncertainty_estimator = (
            uncertainty_estimator or ConfidenceMarginEstimator()
        )

    def calibrate(
        self,
        result: VerificationResult,
        definition: CalibrationDefinition,
        uncertainty_estimator: BaseUncertaintyEstimator | None = None,
    ) -> CalibrationResult:
        p = result.confidence if result.confidence is not None else 0.0
        params: TemperatureScalingParameters = definition.parameters
        t = params.temperature

        p_clipped = max(1e-15, min(1.0 - 1e-15, p))
        logit = math.log(p_clipped / (1.0 - p_clipped))
        scaled_logit = logit / t
        calibrated_p = 1.0 / (1.0 + math.exp(-scaled_logit))

        c_min, c_max = definition.confidence_bounds
        final_p = max(c_min, min(c_max, calibrated_p))

        estimator = uncertainty_estimator or self.uncertainty_estimator
        u = estimator.estimate_uncertainty(final_p)

        trace = CalibrationTrace(
            original_confidence=result.confidence,
            intermediate_values={
                "clipped_confidence": p_clipped,
                "logit": logit,
                "scaled_logit": scaled_logit,
                "unclamped_calibrated_confidence": calibrated_p,
            },
            final_confidence=final_p,
            applied_strategy=CalibrationStrategyType.TEMPERATURE_SCALING,
            parameter_version=definition.calibration_version,
        )
        return CalibrationResult(
            original_confidence=result.confidence,
            calibrated_confidence=final_p,
            uncertainty_estimate=u,
            calibration_trace=trace,
            calibration_metadata={},
        )


class PlattScalingCalibration(BaseCalibrationStrategy):
    """Platt scaling calibration strategy applying sigmoid(slope * logit + intercept)."""

    def __init__(
        self, uncertainty_estimator: BaseUncertaintyEstimator | None = None
    ) -> None:
        self.uncertainty_estimator = (
            uncertainty_estimator or ConfidenceMarginEstimator()
        )

    def calibrate(
        self,
        result: VerificationResult,
        definition: CalibrationDefinition,
        uncertainty_estimator: BaseUncertaintyEstimator | None = None,
    ) -> CalibrationResult:
        p = result.confidence if result.confidence is not None else 0.0
        params: PlattScalingParameters = definition.parameters
        a = params.slope
        b = params.intercept

        p_clipped = max(1e-15, min(1.0 - 1e-15, p))
        logit = math.log(p_clipped / (1.0 - p_clipped))
        scaled_logit = a * logit + b
        calibrated_p = 1.0 / (1.0 + math.exp(-scaled_logit))

        c_min, c_max = definition.confidence_bounds
        final_p = max(c_min, min(c_max, calibrated_p))

        estimator = uncertainty_estimator or self.uncertainty_estimator
        u = estimator.estimate_uncertainty(final_p)

        trace = CalibrationTrace(
            original_confidence=result.confidence,
            intermediate_values={
                "clipped_confidence": p_clipped,
                "logit": logit,
                "scaled_logit": scaled_logit,
                "unclamped_calibrated_confidence": calibrated_p,
            },
            final_confidence=final_p,
            applied_strategy=CalibrationStrategyType.PLATT_SCALING,
            parameter_version=definition.calibration_version,
        )
        return CalibrationResult(
            original_confidence=result.confidence,
            calibrated_confidence=final_p,
            uncertainty_estimate=u,
            calibration_trace=trace,
            calibration_metadata={},
        )


class IsotonicCalibration(BaseCalibrationStrategy):
    """Isotonic calibration strategy performing monotonic piecewise linear interpolation."""

    def __init__(
        self, uncertainty_estimator: BaseUncertaintyEstimator | None = None
    ) -> None:
        self.uncertainty_estimator = (
            uncertainty_estimator or ConfidenceMarginEstimator()
        )

    def calibrate(
        self,
        result: VerificationResult,
        definition: CalibrationDefinition,
        uncertainty_estimator: BaseUncertaintyEstimator | None = None,
    ) -> CalibrationResult:
        p = result.confidence if result.confidence is not None else 0.0
        params: IsotonicCalibrationParameters = definition.parameters
        x_list = params.x_thresholds
        y_list = params.y_values

        if p <= x_list[0]:
            calibrated_p = y_list[0]
            idx = 0
            interp_val = 0.0
        elif p >= x_list[-1]:
            calibrated_p = y_list[-1]
            idx = len(x_list) - 1
            interp_val = 0.0
        else:
            idx = bisect.bisect_right(x_list, p) - 1
            x_low = x_list[idx]
            x_high = x_list[idx + 1]
            y_low = y_list[idx]
            y_high = y_list[idx + 1]

            denom = x_high - x_low
            interp_val = (p - x_low) / denom if denom > 0.0 else 0.0
            calibrated_p = y_low + interp_val * (y_high - y_low)

        c_min, c_max = definition.confidence_bounds
        final_p = max(c_min, min(c_max, calibrated_p))

        estimator = uncertainty_estimator or self.uncertainty_estimator
        u = estimator.estimate_uncertainty(final_p)

        trace = CalibrationTrace(
            original_confidence=result.confidence,
            intermediate_values={
                "matched_bin_index": float(idx),
                "interpolation_factor": interp_val,
                "unclamped_calibrated_confidence": calibrated_p,
            },
            final_confidence=final_p,
            applied_strategy=CalibrationStrategyType.ISOTONIC_CALIBRATION,
            parameter_version=definition.calibration_version,
        )
        return CalibrationResult(
            original_confidence=result.confidence,
            calibrated_confidence=final_p,
            uncertainty_estimate=u,
            calibration_trace=trace,
            calibration_metadata={},
        )
