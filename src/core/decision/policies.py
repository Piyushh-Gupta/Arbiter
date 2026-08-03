"""Concrete decision metric and risk policies and resolver implementations (M4.3 & M4.4)."""

import math
from typing import Any, cast

from src.core.decision.base import BaseDecisionMetricPolicy, BaseRiskPolicy
from src.core.decision.decision_models import (
    DecisionContext,
    DecisionDefinition,
    DecisionMetricPolicyRegistry,
    DecisionMetrics,
    RiskEvaluation,
    RiskTrace,
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


class RawRiskPolicy(BaseRiskPolicy):
    """Produces baseline risk values without failure severity adjustments."""

    @property
    def policy_id(self) -> str:
        return "raw_risk"

    def validate_compatibility(self, definition: Any) -> None:
        """Validates compatibility."""
        pass

    def evaluate_risk(
        self,
        context: DecisionContext,
        metrics: DecisionMetrics,
        definition: DecisionDefinition,
    ) -> RiskEvaluation:
        """Baseline risk mapping: risk_score set directly to uncertainty."""
        trace = RiskTrace(
            factor_id="baseline",
            adjustment_reason="Baseline risk mapping: risk_score set to uncertainty",
            confidence_delta=0.0,
            uncertainty_delta=0.0,
        )

        return RiskEvaluation(
            risk_score=metrics.uncertainty,
            adjusted_confidence=metrics.confidence,
            adjusted_uncertainty=metrics.uncertainty,
            applied_policy_id=self.policy_id,
            contributing_factors=("baseline",),
            risk_traces=(trace,),
        )


class SeverityThresholdRiskPolicy(BaseRiskPolicy):
    """Produces deterministic adjustments based on detected failure severities and flags."""

    @property
    def policy_id(self) -> str:
        return "severity_aware"

    def validate_compatibility(self, definition: Any) -> None:
        """Validates compatibility."""
        pass

    def evaluate_risk(
        self,
        context: DecisionContext,
        metrics: DecisionMetrics,
        definition: DecisionDefinition,
    ) -> RiskEvaluation:
        """Evaluates severity and active failure flags penalties deterministically."""
        traces: list[RiskTrace] = []
        conf_penalty = 0.0
        unc_penalty = 0.0

        # Factor 1: Severity Level Penalty
        severity_val = None
        if context.severity_result and hasattr(
            context.severity_result, "overall_severity"
        ):
            severity_val = getattr(context.severity_result, "overall_severity", None)

        severity_str = (
            str(severity_val.value if hasattr(severity_val, "value") else severity_val)
            if severity_val
            else "NONE"
        )

        sev_mapping = {
            "INFO": (0.0, 0.0),
            "LOW": (0.05, 0.05),
            "MEDIUM": (0.15, 0.15),
            "HIGH": (0.30, 0.30),
            "CRITICAL": (0.50, 0.50),
        }

        sev_conf, sev_unc = sev_mapping.get(severity_str, (0.0, 0.0))
        if sev_conf > 0.0 or sev_unc > 0.0:
            conf_penalty += sev_conf
            unc_penalty += sev_unc
            traces.append(
                RiskTrace(
                    factor_id="severity_penalty",
                    adjustment_reason=f"Severity penalty applied for severity level: {severity_str}",
                    confidence_delta=-sev_conf,
                    uncertainty_delta=sev_unc,
                )
            )

        # Factor 2: Active Failure Flags Penalty
        flags = None
        if context.failure_analysis_result and hasattr(
            context.failure_analysis_result, "failure_flags"
        ):
            flags = getattr(context.failure_analysis_result, "failure_flags", None)

        if flags:
            flag_count = len(flags)
            flag_conf = min(0.20, 0.05 * flag_count)
            flag_unc = min(0.20, 0.05 * flag_count)
            conf_penalty += flag_conf
            unc_penalty += flag_unc
            traces.append(
                RiskTrace(
                    factor_id="failure_flags_penalty",
                    adjustment_reason=f"Failure flags penalty applied for flags: {list(flags)}",
                    confidence_delta=-flag_conf,
                    uncertainty_delta=flag_unc,
                )
            )

        adjusted_conf = max(0.0, min(1.0, metrics.confidence - conf_penalty))
        adjusted_unc = max(0.0, min(1.0, metrics.uncertainty + unc_penalty))

        contributing = tuple(t.factor_id for t in traces)

        return RiskEvaluation(
            risk_score=adjusted_unc,
            adjusted_confidence=adjusted_conf,
            adjusted_uncertainty=adjusted_unc,
            applied_policy_id=self.policy_id,
            contributing_factors=contributing,
            risk_traces=tuple(traces),
        )


class CostBenefitRiskPolicy(BaseRiskPolicy):
    """Produces deterministic adjustments based on expected cost-benefit utility assessments."""

    @property
    def policy_id(self) -> str:
        return "cost_benefit"

    def validate_compatibility(self, definition: Any) -> None:
        """Validates compatibility."""
        pass

    def evaluate_risk(
        self,
        context: DecisionContext,
        metrics: DecisionMetrics,
        definition: DecisionDefinition,
    ) -> RiskEvaluation:
        """Utility cost-benefit analysis computing risk score and adjusting metrics."""
        traces: list[RiskTrace] = []
        conf_penalty = 0.0
        unc_penalty = 0.0

        # Factor 1: Uncertainty Risk Penalty
        if metrics.uncertainty > 0.4:
            penalty = 0.10
            conf_penalty += penalty
            unc_penalty += penalty
            traces.append(
                RiskTrace(
                    factor_id="uncertainty_risk",
                    adjustment_reason=f"Uncertainty exceeds threshold (score: {metrics.uncertainty:.3f} > 0.40)",
                    confidence_delta=-penalty,
                    uncertainty_delta=penalty,
                )
            )

        # Factor 2: Operational Error Risk Penalty
        has_errors = False
        if context.failure_analysis_result and hasattr(
            context.failure_analysis_result, "failure_flags"
        ):
            flags = getattr(context.failure_analysis_result, "failure_flags", None)
            if flags and len(flags) > 0:
                has_errors = True

        if has_errors:
            penalty = 0.20
            conf_penalty += penalty
            unc_penalty += penalty
            traces.append(
                RiskTrace(
                    factor_id="error_risk",
                    adjustment_reason="Operational errors present in failure analysis pipeline",
                    confidence_delta=-penalty,
                    uncertainty_delta=penalty,
                )
            )

        adjusted_conf = max(0.0, min(1.0, metrics.confidence - conf_penalty))
        adjusted_unc = max(0.0, min(1.0, metrics.uncertainty + unc_penalty))

        contributing = tuple(t.factor_id for t in traces)

        return RiskEvaluation(
            risk_score=adjusted_unc,
            adjusted_confidence=adjusted_conf,
            adjusted_uncertainty=adjusted_unc,
            applied_policy_id=self.policy_id,
            contributing_factors=contributing,
            risk_traces=tuple(traces),
        )
