"""Verification Failure Analysis subsystem (M3.5)."""

from src.core.failure.attribution import (
    BaseRootCauseStrategy,
    DependencyGraphRootCauseStrategy,
)
from src.core.failure.base import BaseFailureAnalyzer, FailureAggregationStrategy
from src.core.failure.correlation import (
    BaseFailureCorrelationStrategy,
    DefaultFailureCorrelationStrategy,
)
from src.core.failure.failure_models import (
    AnalyzerExecutionResult,
    DiagnosticEvidence,
    FailureAnalysisDefinition,
    FailureAnalysisInput,
    FailureAnalysisProfile,
    FailureAnalysisProfileRegistry,
    FailureAnalysisResult,
    FailureArtifactReference,
    FailureCategory,
    FailureClassification,
    FailureCorrelation,
    FailureCorrelationContext,
    FailureCorrelationDefinition,
    FailureCorrelationProfile,
    FailureCorrelationProfileRegistry,
    FailureCorrelationResult,
    FailureCorrelationRule,
    FailureDiagnostic,
    FailureDiagnosticContext,
    FailureExecutionMetadata,
    FailureRootCause,
    FailureRuntimeMetadata,
    FailureSeverity,
    FailureTrace,
    RootCauseAttributionDefinition,
    RootCauseProfile,
    RootCauseProfileRegistry,
    RootCauseResult,
    SeverityEvaluationResult,
    SeverityPolicyDefinition,
    SeverityPolicyProfile,
    SeverityPolicyRegistry,
    SeverityRule,
)
from src.core.failure.implementations import (
    CalibrationFailureAnalyzer,
    CompositeFailureAnalyzer,
    DefaultFailureAggregationStrategy,
    DefaultFailureAnalyzer,
    InfrastructureFailureAnalyzer,
    RetrievalFailureAnalyzer,
    VerificationFailureAnalyzer,
)
from src.core.failure.severity import BaseSeverityPolicy, ThresholdSeverityPolicy
from src.core.failure.traversal import FailureGraphTraverser

__all__ = [
    # Enums & primitives
    "FailureSeverity",
    "FailureCategory",
    "FailureRootCause",
    "FailureClassification",
    "FailureDiagnostic",
    "FailureTrace",
    # Analysis definitions & results
    "FailureAnalysisDefinition",
    "FailureAnalysisResult",
    "FailureAnalysisProfile",
    "FailureAnalysisProfileRegistry",
    "FailureAnalysisInput",
    "FailureArtifactReference",
    "FailureRuntimeMetadata",
    "FailureExecutionMetadata",
    "DiagnosticEvidence",
    "AnalyzerExecutionResult",
    "FailureDiagnosticContext",
    # Analyzer protocols & implementations
    "BaseFailureAnalyzer",
    "DefaultFailureAnalyzer",
    "FailureAggregationStrategy",
    "RetrievalFailureAnalyzer",
    "VerificationFailureAnalyzer",
    "CalibrationFailureAnalyzer",
    "InfrastructureFailureAnalyzer",
    "DefaultFailureAggregationStrategy",
    "CompositeFailureAnalyzer",
    # Correlation models
    "FailureCorrelationDefinition",
    "FailureCorrelationRule",
    "FailureCorrelation",
    "FailureCorrelationContext",
    "FailureCorrelationResult",
    "FailureCorrelationProfile",
    "FailureCorrelationProfileRegistry",
    "BaseFailureCorrelationStrategy",
    "DefaultFailureCorrelationStrategy",
    # Traversal
    "FailureGraphTraverser",
    # Root cause attribution
    "RootCauseAttributionDefinition",
    "RootCauseResult",
    "RootCauseProfile",
    "RootCauseProfileRegistry",
    "BaseRootCauseStrategy",
    "DependencyGraphRootCauseStrategy",
    # Severity policy
    "SeverityRule",
    "SeverityPolicyDefinition",
    "SeverityEvaluationResult",
    "SeverityPolicyProfile",
    "SeverityPolicyRegistry",
    "BaseSeverityPolicy",
    "ThresholdSeverityPolicy",
]
