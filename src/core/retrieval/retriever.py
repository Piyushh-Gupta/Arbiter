"""Stateless orchestrator for evidence retrieval."""

from src.core.retrieval.base import BaseRetriever
from src.core.retrieval.retrieval_models import EvidenceBundle, RetrievalDefinition


class ClaimRetriever:
    """
    Instance-based pure orchestrator for the retrieval subsystem.
    """

    def retrieve(
        self,
        claim: str,
        definition: RetrievalDefinition,
        strategy: BaseRetriever,
    ) -> EvidenceBundle:
        """
        Orchestrates the execution of a single retrieval strategy.

        Execution Semantics:
        1. Assumes definition and strategy are structurally compatible.
        2. Injects the claim and definition into strategy.retrieve().
        3. Returns the exact EvidenceBundle produced by the strategy without modification.
        """
        return strategy.retrieve(claim, definition)
