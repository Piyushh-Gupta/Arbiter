"""Concrete decision metric policies and resolver implementations (M4.3)."""

import math
from typing import Any, cast

from src.core.decision.base import BaseDecisionMetricPolicy
from src.core.decision.decision_models import (
    DecisionContext,
    DecisionDefinition,
    DecisionMetricPolicyRegistry,
    DecisionMetrics,
)
from src.core.exceptions import DecisionExecutionError


class CalibratedMetricPolicy(BaseDecisionMetricPolicy):
    """Computes confidence and uncertainty using calibrated confidence scores."""

    @property
    def policy_id(self) -> str:
        return "calibrated"

    def validate_compatibility(self, definition: Any) -> None:
        """Validates that definition is compatible."""
        pass

    def evaluate_metrics(
        self, context: DecisionContext, definition: Any
    ) -> DecisionMetrics:
        """Computes metrics from calibrated confidence result."""
        if not context.calibration_result:
            raise DecisionExecutionError("Calibration result is missing.")

        calibrated_conf = getattr(
            context.calibration_result, "calibrated_confidence", None
        )
        if calibrated_conf is None:
            raise DecisionExecutionError(
                "Calibrated confidence score is missing from calibration result."
            )

        return DecisionMetrics(
            confidence=float(calibrated_conf),
            uncertainty=1.0 - float(calibrated_conf),
            calibrated=True,
            source="calibration",
            metadata={"policy_id": self.policy_id},
        )


class RawMetricPolicy(BaseDecisionMetricPolicy):
    """Computes confidence and uncertainty from raw verification output."""

    @property
    def policy_id(self) -> str:
        return "raw"

    def validate_compatibility(self, definition: Any) -> None:
        """Validates that definition is compatible."""
        pass

    def evaluate_metrics(
        self, context: DecisionContext, definition: Any
    ) -> DecisionMetrics:
        """Computes metrics from raw verification result."""
        if not context.verification_result:
            raise DecisionExecutionError("Verification result is missing.")

        raw_conf = getattr(context.verification_result, "confidence", None)
        if raw_conf is None:
            raise DecisionExecutionError(
                "Confidence score is missing from verification result."
            )

        return DecisionMetrics(
            confidence=float(raw_conf),
            uncertainty=1.0 - float(raw_conf),
            calibrated=False,
            source="verification",
            metadata={"policy_id": self.policy_id},
        )


class EntropyMetricPolicy(BaseDecisionMetricPolicy):
    """Computes uncertainty using normalized binary Shannon entropy."""

    @property
    def policy_id(self) -> str:
        return "entropy"

    def validate_compatibility(self, definition: Any) -> None:
        """Validates that definition is compatible."""
        pass

    def evaluate_metrics(
        self, context: DecisionContext, definition: Any
    ) -> DecisionMetrics:
        """Computes metrics using binary Shannon entropy over calibrated confidence."""
        if not context.calibration_result:
            raise DecisionExecutionError("Calibration result is missing.")

        calibrated_conf = getattr(
            context.calibration_result, "calibrated_confidence", None
        )
        if calibrated_conf is None:
            raise DecisionExecutionError(
                "Calibrated confidence score is missing from calibration result."
            )

        p = max(1e-15, min(1.0 - 1e-15, float(calibrated_conf)))
        entropy_uncertainty = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

        return DecisionMetrics(
            confidence=float(calibrated_conf),
            uncertainty=float(entropy_uncertainty),
            calibrated=True,
            source="entropy",
            metadata={"policy_id": self.policy_id},
        )


class DecisionMetricResolver:
    """Resolves decision metrics using a deterministic fallback order: Calibration -> Verification -> Default."""

    def __init__(
        self,
        registry: DecisionMetricPolicyRegistry,
        default_confidence: float = 0.5,
        default_uncertainty: float = 0.5,
    ) -> None:
        self.registry = registry
        self.default_confidence = default_confidence
        self.default_uncertainty = default_uncertainty

    def resolve_metrics(
        self, context: DecisionContext, definition: DecisionDefinition
    ) -> DecisionMetrics:
        """
        Attempts to resolve and compute metrics in order:
        1. Configured policy (specified by definition.confidence_policy)
        2. Verification fallback (using 'raw' metric policy)
        3. Configured default fallback metrics
        """
        # 1. Configured metric policy
        policy_id = getattr(definition, "confidence_policy", "calibrated")
        try:
            policy = self.registry.resolve(policy_id)
            return cast(DecisionMetrics, policy.evaluate_metrics(context, definition))
        except Exception:
            pass

        # 2. Verification fallback
        try:
            raw_policy = self.registry.resolve("raw")
            return cast(
                DecisionMetrics, raw_policy.evaluate_metrics(context, definition)
            )
        except Exception:
            pass

        # 3. Default fallback
        return DecisionMetrics(
            confidence=self.default_confidence,
            uncertainty=self.default_uncertainty,
            calibrated=False,
            source="default",
            metadata={"fallback": True},
        )
