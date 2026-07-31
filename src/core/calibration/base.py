"""Stateless protocols for the Calibration subsystem."""

from typing import Protocol, runtime_checkable

from src.core.calibration.calibration_models import (
    CalibrationDefinition,
    CalibrationResult,
)
from src.core.verification.verification_models import VerificationResult


@runtime_checkable
class BaseCalibrationStrategy(Protocol):
    """Protocol for post-verification confidence calibration strategies."""

    def calibrate(
        self,
        result: VerificationResult,
        definition: CalibrationDefinition,
    ) -> CalibrationResult:
        """
        Calibrates the confidence of a verification result.

        Args:
            result: Uncalibrated VerificationResult.
            definition: Calibration configuration definition.

        Returns:
            CalibrationResult: Immutable calibrated confidence and metadata.
        """
        ...


@runtime_checkable
class BaseUncertaintyEstimator(Protocol):
    """Protocol for uncertainty estimation models."""

    def estimate_uncertainty(self, calibrated_confidence: float) -> float:
        """
        Computes an uncertainty estimate in [0, 1] based on calibrated confidence.

        Args:
            calibrated_confidence: Calibrated probability value in [0, 1].

        Returns:
            float: Deterministic uncertainty estimate.
        """
        ...
