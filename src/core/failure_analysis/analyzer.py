"""Stateless orchestrator for failure analysis."""

from src.core.verification.verification_models import VerificationResult

from .base import BaseFailureAnalyzer
from .failure_analysis_models import FailureAnalysisDefinition, FailureAnalysisResult


class FailureAnalyzer:
    """
    Instance-based pure orchestrator for the failure analysis subsystem.
    """

    def analyze(
        self,
        claim: str,
        verification_result: VerificationResult,
        definition: FailureAnalysisDefinition,
        strategy: BaseFailureAnalyzer,
    ) -> FailureAnalysisResult:
        """
        Orchestrates the execution of a single failure analysis strategy.

        Execution Semantics:
        1. Assumes definition and strategy are structurally compatible.
        2. Injects claim, verification_result, and definition into strategy.analyze().
        3. Returns the exact FailureAnalysisResult produced by the strategy without modification.
        """
        return strategy.analyze(claim, verification_result, definition)
