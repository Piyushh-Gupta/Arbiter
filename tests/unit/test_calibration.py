"""Unit and integration tests for the confidence calibration and uncertainty estimation subsystem."""

import math

import pytest
from pydantic import ValidationError

from src.core.bootstrap import build_calibration_registry
from src.core.calibration.calibration_models import (
    CalibrationDefinition,
    CalibrationStrategyType,
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
from src.core.config import Settings
from src.core.exceptions import CalibrationConfigurationError
from src.core.verification.verification_models import (
    VerificationResult,
    VerificationVerdict,
)


@pytest.fixture
def mock_verification_result() -> VerificationResult:
    return VerificationResult(
        verdict=VerificationVerdict.SUPPORTED,
        confidence=0.8,
        supporting_passages=("s1",),
        contradicting_passages=(),
    )


def test_identity_calibration(mock_verification_result: VerificationResult) -> None:
    strategy = IdentityCalibration()
    definition = CalibrationDefinition(
        strategy=CalibrationStrategyType.IDENTITY,
        parameters=None,
    )
    res = strategy.calibrate(mock_verification_result, definition)
    assert res.original_confidence == 0.8
    assert res.calibrated_confidence == 0.8
    assert res.uncertainty_estimate == pytest.approx(0.2)  # default margin
    assert res.calibration_trace.applied_strategy == CalibrationStrategyType.IDENTITY


def test_temperature_scaling_calibration(
    mock_verification_result: VerificationResult,
) -> None:
    # T = 2.0.
    # logit for 0.8 = log(0.8 / 0.2) = log(4) = 1.38629436
    # scaled logit = log(4) / 2.0 = 0.69314718
    # calibrated p = sigmoid(0.69314718) = 1.0 / (1.0 + exp(-0.69314718)) = 1.0 / (1.0 + 0.5) = 2/3 = 0.6666666
    strategy = TemperatureScalingCalibration()
    params = TemperatureScalingParameters(temperature=2.0)
    definition = CalibrationDefinition(
        strategy=CalibrationStrategyType.TEMPERATURE_SCALING,
        parameters=params,
    )
    res = strategy.calibrate(mock_verification_result, definition)
    assert res.original_confidence == 0.8
    assert pytest.approx(res.calibrated_confidence) == 2.0 / 3.0
    assert res.calibration_trace.intermediate_values["logit"] == pytest.approx(
        math.log(4.0)
    )


def test_platt_scaling_calibration(
    mock_verification_result: VerificationResult,
) -> None:
    # slope = 0.5, intercept = 0.1
    # logit for 0.8 = log(4) = 1.38629436
    # scaled logit = 0.5 * log(4) + 0.1 = 0.69314718 + 0.1 = 0.79314718
    # calibrated p = sigmoid(0.79314718) = 0.68849767
    strategy = PlattScalingCalibration()
    params = PlattScalingParameters(slope=0.5, intercept=0.1)
    definition = CalibrationDefinition(
        strategy=CalibrationStrategyType.PLATT_SCALING,
        parameters=params,
    )
    res = strategy.calibrate(mock_verification_result, definition)
    assert res.original_confidence == 0.8
    expected = 1.0 / (1.0 + math.exp(-(0.5 * math.log(4.0) + 0.1)))
    assert pytest.approx(res.calibrated_confidence) == expected


def test_isotonic_calibration(mock_verification_result: VerificationResult) -> None:
    # Interpolation case: p = 0.8
    # Bins: [0.0, 0.5, 1.0] -> [0.0, 0.6, 1.0]
    # idx = 1 (x_low = 0.5, x_high = 1.0, y_low = 0.6, y_high = 1.0)
    # interp factor = (0.8 - 0.5) / (1.0 - 0.5) = 0.3 / 0.5 = 0.6
    # calibrated p = 0.6 + 0.6 * (1.0 - 0.6) = 0.6 + 0.24 = 0.84
    strategy = IsotonicCalibration()
    params = IsotonicCalibrationParameters(
        x_thresholds=(0.0, 0.5, 1.0),
        y_values=(0.0, 0.6, 1.0),
    )
    definition = CalibrationDefinition(
        strategy=CalibrationStrategyType.ISOTONIC_CALIBRATION,
        parameters=params,
    )
    res = strategy.calibrate(mock_verification_result, definition)
    assert res.original_confidence == 0.8
    assert pytest.approx(res.calibrated_confidence) == 0.84

    # Boundary handling: p <= x[0]
    res_low = strategy.calibrate(
        VerificationResult(verdict=VerificationVerdict.SUPPORTED, confidence=0.0),
        definition,
    )
    assert res_low.calibrated_confidence == 0.0

    # Boundary handling: p >= x[-1]
    res_high = strategy.calibrate(
        VerificationResult(verdict=VerificationVerdict.SUPPORTED, confidence=1.0),
        definition,
    )
    assert res_high.calibrated_confidence == 1.0


def test_uncertainty_estimators() -> None:
    # Margin
    margin = ConfidenceMarginEstimator()
    assert margin.estimate_uncertainty(0.8) == pytest.approx(0.2)

    # Entropy
    entropy = EntropyEstimator()
    # H(0.5) = - (0.5 * log2(0.5) + 0.5 * log2(0.5)) = 1.0
    assert entropy.estimate_uncertainty(0.5) == pytest.approx(1.0)
    # H(0.8) = - (0.8 * log2(0.8) + 0.2 * log2(0.2)) = 0.72192809
    assert pytest.approx(entropy.estimate_uncertainty(0.8)) == 0.7219280948

    # Variance
    variance = NormalizedVarianceEstimator()
    # 4 * 0.5 * 0.5 = 1.0
    assert variance.estimate_uncertainty(0.5) == pytest.approx(1.0)
    # 4 * 0.8 * 0.2 = 0.64
    assert variance.estimate_uncertainty(0.8) == pytest.approx(0.64)


def test_monotonicity_validation() -> None:
    # Monotonicity failure
    with pytest.raises(ValidationError):
        IsotonicCalibrationParameters(
            x_thresholds=(0.0, 0.5, 1.0),
            y_values=(0.0, 0.4, 0.3),  # Descending
        )

    # Threshold length mismatch
    with pytest.raises(ValidationError):
        IsotonicCalibrationParameters(
            x_thresholds=(0.0, 0.5),
            y_values=(0.0, 0.5, 1.0),
        )

    # Temperature validation
    with pytest.raises(ValidationError):
        TemperatureScalingParameters(temperature=0.0)
    with pytest.raises(ValidationError):
        TemperatureScalingParameters(temperature=-1.0)


def test_registry_and_bootstrap() -> None:
    settings = Settings()
    settings.calibration.temperature = 1.5
    registry = build_calibration_registry(settings)

    assert registry.resolve("identity") is not None
    assert registry.resolve("temperature_scaling") is not None
    assert registry.resolve("platt_scaling") is not None
    assert registry.resolve("isotonic_calibration") is not None

    # Duplicate resolution keys / invalid temperatures validation checks
    settings.calibration.temperature = -0.5
    with pytest.raises(CalibrationConfigurationError):
        build_calibration_registry(settings)


def test_determinism(mock_verification_result: VerificationResult) -> None:
    strategy = TemperatureScalingCalibration()
    params = TemperatureScalingParameters(temperature=1.5)
    definition = CalibrationDefinition(
        strategy=CalibrationStrategyType.TEMPERATURE_SCALING,
        parameters=params,
    )
    res1 = strategy.calibrate(mock_verification_result, definition)
    res2 = strategy.calibrate(mock_verification_result, definition)
    assert res1.calibrated_confidence == res2.calibrated_confidence
    assert res1.uncertainty_estimate == res2.uncertainty_estimate
    assert (
        res1.calibration_trace.intermediate_values
        == res2.calibration_trace.intermediate_values
    )
