"""Stateless base verification protocol."""

from typing import Protocol, runtime_checkable

from src.core.retrieval.retrieval_models import EvidenceBundle
from src.core.verification.verification_models import (
    VerificationDefinition,
    VerificationResult,
)


@runtime_checkable
class BaseVerifier(Protocol):
    """Stateless protocol for all verification strategies."""

    def validate_compatibility(self, definition: VerificationDefinition) -> None:
        """
        Statically verifies if this verifier supports the given definition.
        Raises VerificationConfigurationError if incompatible.
        Must not perform any I/O or model inference.
        """
        ...

    def verify(
        self,
        claim: str,
        bundle: EvidenceBundle,
        definition: VerificationDefinition,
    ) -> VerificationResult:
        """
        Executes verification logic.

        Receives:
        - claim: The normalized, verified textual assertion.
        - bundle: The immutable, ordered collection of evidence passages.
        - definition: The validated, immutable configuration parameters.

        Returns:
        - VerificationResult: A fully materialized, immutable verdict.

        Must not modify the bundle.
        Must not perform filesystem or network access.
        Must not cache internal state.
        """
        ...
