"""Stateless orchestrator for uncertainty estimation."""

from src.core.failure_analysis.failure_analysis_models import FailureAnalysisResult
from src.core.uncertainty.base import BaseUncertaintyEstimator
from src.core.uncertainty.uncertainty_models import (
    UncertaintyDefinition,
    UncertaintyResult,
)


class UncertaintyEstimator:
    """
    Instance-based pure orchestrator for the uncertainty estimation subsystem.
    """

    def estimate(
        self,
        claim: str,
        failure_analysis_result: FailureAnalysisResult,
        definition: UncertaintyDefinition,
        strategy: BaseUncertaintyEstimator,
    ) -> UncertaintyResult:
        """
        Orchestrates the execution of a single uncertainty estimation strategy.

        Execution Semantics:
        1. Assumes definition and strategy are structurally compatible.
        2. Injects inputs into strategy.estimate().
        3. Returns the exact UncertaintyResult produced by the strategy without modification.
        4. Does NOT catch exceptions. The strategy must raise UncertaintyExecutionError directly.
        """
        return strategy.estimate(claim, failure_analysis_result, definition)
