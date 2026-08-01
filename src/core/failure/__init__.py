"""Verification Failure Analysis subsystem (M3.1)."""

from src.core.failure.base import BaseFailureAnalyzer
from src.core.failure.failure_models import (
    FailureAnalysisDefinition,
    FailureAnalysisProfile,
    FailureAnalysisProfileRegistry,
    FailureAnalysisResult,
    FailureCategory,
    FailureClassification,
    FailureDiagnostic,
    FailureRootCause,
    FailureSeverity,
    FailureTrace,
)
from src.core.failure.implementations import DefaultFailureAnalyzer

__all__ = [
    "FailureSeverity",
    "FailureCategory",
    "FailureRootCause",
    "FailureClassification",
    "FailureDiagnostic",
    "FailureTrace",
    "FailureAnalysisDefinition",
    "FailureAnalysisResult",
    "FailureAnalysisProfile",
    "FailureAnalysisProfileRegistry",
    "BaseFailureAnalyzer",
    "DefaultFailureAnalyzer",
]
