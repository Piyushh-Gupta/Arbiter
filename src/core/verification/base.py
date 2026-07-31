"""Stateless base verification protocol."""

from typing import Protocol, runtime_checkable

from src.core.retrieval.retrieval_models import EvidenceBundle
from src.core.verification.verification_models import (
    PassageVerificationResult,
    VerificationDefinition,
    VerificationResult,
    VerifierRuntimeMetadata,
)


@runtime_checkable
class BaseMetadataProvider(Protocol):
    """Protocol for verification metadata providers."""

    def get_runtime_metadata(self) -> VerifierRuntimeMetadata:
        """
        Retrieves the current verifier runtime metadata.

        Returns:
            VerifierRuntimeMetadata: Timezone-aware execution and system details.
        """
        ...


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

    def verify_passages(
        self,
        claim: str,
        bundle: EvidenceBundle,
    ) -> tuple[PassageVerificationResult, ...]:
        """
        Executes passage-level verification logic independently on evidence passages.

        Receives:
        - claim: The normalized textual assertion.
        - bundle: The immutable collection of evidence passages.

        Returns:
        - tuple[PassageVerificationResult, ...]: Passage-level outcomes.
        """
        return ()

    def verify(
        self,
        claim: str,
        bundle: EvidenceBundle,
        definition: VerificationDefinition,
    ) -> VerificationResult:
        """
        Executes full verification logic and produces claim-level verification result.

        Receives:
        - claim: The normalized textual assertion.
        - bundle: The immutable collection of evidence passages.
        - definition: Validated configuration definition.

        Returns:
        - VerificationResult: Fully materialized immutable verdict.
        """
        ...
