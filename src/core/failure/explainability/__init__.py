"""Failure Explainability & Reporting subsystem (M3.7)."""

from src.core.failure.explainability.base import BaseFailureExplanationStrategy
from src.core.failure.explainability.explanation_models import (
    FailureDecisionTrace,
    FailureEvidenceExplanation,
    FailureExplanationDefinition,
    FailureExplanationMetadata,
    FailureExplanationProfile,
    FailureExplanationProfileRegistry,
    FailureExplanationResult,
    FailureExplanationTemplate,
    compute_explanation_fingerprint,
)
from src.core.failure.explainability.implementations import (
    CompositeFailureExplanationStrategy,
    DecisionTraceExplanationStrategy,
    SummaryExplanationStrategy,
)
from src.core.failure.explainability.rendering import FailureReportRenderer

__all__ = [
    "BaseFailureExplanationStrategy",
    "CompositeFailureExplanationStrategy",
    "DecisionTraceExplanationStrategy",
    "FailureDecisionTrace",
    "FailureEvidenceExplanation",
    "FailureExplanationDefinition",
    "FailureExplanationMetadata",
    "FailureExplanationProfile",
    "FailureExplanationProfileRegistry",
    "FailureExplanationResult",
    "FailureExplanationTemplate",
    "FailureReportRenderer",
    "SummaryExplanationStrategy",
    "compute_explanation_fingerprint",
]
