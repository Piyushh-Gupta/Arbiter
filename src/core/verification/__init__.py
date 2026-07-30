"""Verification subsystem for Arbiter."""

from src.core.verification.aggregation import (
    BaseAggregationStrategy,
    MaxConfidenceAggregationStrategy,
)
from src.core.verification.base import BaseVerifier
from src.core.verification.implementations import NLIVerifier
from src.core.verification.verification_models import (
    NLIVerificationDefinition,
    PassageVerificationResult,
    VerificationDefinition,
    VerificationExplanation,
    VerificationLabel,
    VerificationMetadata,
    VerificationModelMetadata,
    VerificationProfile,
    VerificationProfileRegistry,
    VerificationResult,
    VerificationVerdict,
    VerifiedPassage,
)

__all__ = [
    "BaseAggregationStrategy",
    "BaseVerifier",
    "MaxConfidenceAggregationStrategy",
    "NLIVerificationDefinition",
    "NLIVerifier",
    "PassageVerificationResult",
    "VerificationDefinition",
    "VerificationExplanation",
    "VerificationLabel",
    "VerificationMetadata",
    "VerificationModelMetadata",
    "VerificationProfile",
    "VerificationProfileRegistry",
    "VerificationResult",
    "VerificationVerdict",
    "VerifiedPassage",
]
