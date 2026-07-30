"""Comprehensive unit tests for M2.1 Verification Architecture Modernization."""

import pytest
from pydantic import ValidationError

from src.core.bootstrap import DummyNLIModel, build_verification_registry
from src.core.config import Settings
from src.core.exceptions import (
    DuplicateVerificationProfileError,
    VerificationProfileNotFoundError,
)
from src.core.retrieval.retrieval_models import (
    EvidenceBundle,
    EvidencePassage,
    RetrievalMetadata,
)
from src.core.verification.aggregation import MaxConfidenceAggregationStrategy
from src.core.verification.implementations import NLIVerifier
from src.core.verification.verification_models import (
    NLIVerificationDefinition,
    PassageVerificationResult,
    VerificationDefinition,
    VerificationModelMetadata,
    VerificationProfile,
    VerificationProfileRegistry,
    VerificationVerdict,
)


@pytest.fixture
def sample_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        claim="Water boils at 100 degrees Celsius at sea level.",
        passages=(
            EvidencePassage(
                document_id="doc1",
                span_id="span-1",
                text="Pure water boils at 100 °C under standard atmospheric pressure.",
                score=0.95,
            ),
            EvidencePassage(
                document_id="doc2",
                span_id="span-2",
                text="Water freezes at 0 degrees Celsius.",
                score=0.85,
            ),
        ),
        metadata=RetrievalMetadata(strategy_id="test", top_k=2),
    )


def test_verification_verdict_enum() -> None:
    assert VerificationVerdict.SUPPORTED.value == "SUPPORTED"
    assert VerificationVerdict.CONTRADICTED.value == "CONTRADICTED"
    assert VerificationVerdict.INSUFFICIENT.value == "INSUFFICIENT"


def test_passage_verification_result_immutability() -> None:
    p_res = PassageVerificationResult(
        span_id="span-1",
        verdict=VerificationVerdict.SUPPORTED,
        confidence=0.92,
        raw_scores={"SUPPORTED": 0.92, "CONTRADICTED": 0.05, "INSUFFICIENT": 0.03},
        rationale="Direct text entailment.",
    )
    assert p_res.span_id == "span-1"
    assert p_res.verdict == VerificationVerdict.SUPPORTED
    assert p_res.confidence == 0.92

    with pytest.raises(ValidationError):
        p_res.confidence = 1.5


def test_max_confidence_aggregation_strategy() -> None:
    strategy = MaxConfidenceAggregationStrategy()
    definition = VerificationDefinition(
        confidence_thresholds={"SUPPORTED": 0.7, "CONTRADICTED": 0.7}
    )
    metadata = VerificationModelMetadata(model_identifier="test-model")

    passage_results = (
        PassageVerificationResult(
            span_id="span-1",
            verdict=VerificationVerdict.SUPPORTED,
            confidence=0.88,
        ),
        PassageVerificationResult(
            span_id="span-2",
            verdict=VerificationVerdict.CONTRADICTED,
            confidence=0.40,
        ),
    )

    result = strategy.aggregate(passage_results, definition, metadata)
    assert result.verdict == VerificationVerdict.SUPPORTED
    assert result.confidence == 0.88
    assert result.supporting_passages == ("span-1",)
    assert result.contradicting_passages == ("span-2",)
    assert result.evidence_attribution["span-1"] == 0.88
    assert result.model_metadata.model_identifier == "test-model"


def test_max_confidence_aggregation_insufficient_threshold() -> None:
    strategy = MaxConfidenceAggregationStrategy()
    definition = VerificationDefinition(
        confidence_thresholds={"SUPPORTED": 0.95, "CONTRADICTED": 0.95}
    )

    passage_results = (
        PassageVerificationResult(
            span_id="span-1",
            verdict=VerificationVerdict.SUPPORTED,
            confidence=0.80,
        ),
    )

    result = strategy.aggregate(passage_results, definition)
    assert result.verdict == VerificationVerdict.INSUFFICIENT
    assert result.confidence == 0.80


def test_nli_verifier_determinism_and_pipeline(sample_bundle: EvidenceBundle) -> None:
    verifier = NLIVerifier(model=DummyNLIModel(), strategy_id="dummy_nli")
    definition = NLIVerificationDefinition(top_k=2)

    verifier.validate_compatibility(definition)

    res1 = verifier.verify(
        "Water boils at 100 degrees Celsius", sample_bundle, definition
    )
    res2 = verifier.verify(
        "Water boils at 100 degrees Celsius", sample_bundle, definition
    )

    # Determinism Assertion
    assert res1 == res2
    assert res1.verdict == VerificationVerdict.SUPPORTED
    assert res1.confidence == 1.0
    assert "span-1" in res1.evidence_attribution


def test_verification_profile_registry() -> None:
    verifier = NLIVerifier(model=DummyNLIModel(), strategy_id="dummy_nli")
    definition = NLIVerificationDefinition(top_k=5)
    profile = VerificationProfile(
        profile_id="test_verifier",
        definition=definition,
        verifier=verifier,
    )

    registry = VerificationProfileRegistry(profiles=(profile,))
    assert registry.resolve("test_verifier") is profile

    with pytest.raises(DuplicateVerificationProfileError):
        VerificationProfileRegistry(profiles=(profile, profile))

    with pytest.raises(VerificationProfileNotFoundError):
        registry.resolve("non_existent")


def test_bootstrap_verification_registry() -> None:
    config = Settings()
    registry = build_verification_registry(config)
    resolved = registry.resolve("default_verification")
    assert resolved.profile_id == "default_verification"
