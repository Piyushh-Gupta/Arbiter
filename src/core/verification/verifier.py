"""Stateless orchestrator for evidence verification."""

from src.core.retrieval.retrieval_models import EvidenceBundle
from src.core.verification.base import BaseVerifier
from src.core.verification.verification_models import (
    VerificationDefinition,
    VerificationResult,
)


class ClaimVerifier:
    """
    Instance-based pure orchestrator for the verification subsystem.
    """

    def verify(
        self,
        claim: str,
        bundle: EvidenceBundle,
        definition: VerificationDefinition,
        strategy: BaseVerifier,
    ) -> VerificationResult:
        """
        Orchestrates the execution of a single verification strategy.

        Execution Semantics:
        1. Assumes definition and strategy are structurally compatible.
        2. Injects claim, bundle, and definition into strategy.verify().
        3. Returns the exact VerificationResult produced by the strategy without modification.
        """
        return strategy.verify(claim, bundle, definition)
