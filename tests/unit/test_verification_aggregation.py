"""Unit tests for multi-evidence aggregation strategies and weigher layers."""

import pytest
from pydantic import ValidationError

from src.core.retrieval.retrieval_models import (
    EvidenceBundle,
    EvidencePassage,
    RetrievalMetadata,
)
from src.core.verification.aggregation import (
    DefaultEvidenceWeigher,
    MaxConfidenceAggregationStrategy,
    sort_passage_results,
)
from src.core.verification.aggregation_strategies import (
    ConsensusAggregationStrategy,
    ContradictionAwareAggregationStrategy,
    WeightedVotingAggregationStrategy,
)
from src.core.verification.verification_models import (
    AggregationProfile,
    AggregationProfileRegistry,
    AggregationStrategyType,
    ClaimVerificationInput,
    NLIVerificationDefinition,
    PassageVerificationResult,
    PassageVerificationScore,
    VerificationVerdict,
)


@pytest.fixture
def test_bundle() -> EvidenceBundle:
    p1 = EvidencePassage(
        document_id="doc1", span_id="s1", text="text1", score=0.8, metadata={}
    )
    p2 = EvidencePassage(
        document_id="doc2", span_id="s2", text="text2", score=0.6, metadata={}
    )
    p3 = EvidencePassage(
        document_id="doc3", span_id="s3", text="text3", score=0.4, metadata={}
    )
    return EvidenceBundle(
        claim="Water boils at 100C",
        passages=(p1, p2, p3),
        metadata=RetrievalMetadata(strategy_id="test", top_k=3),
    )


@pytest.fixture
def mock_passage_results() -> tuple[PassageVerificationResult, ...]:
    pr1 = PassageVerificationResult(
        span_id="s1",
        verdict=VerificationVerdict.SUPPORTED,
        confidence=0.9,
        probability_distribution=PassageVerificationScore(
            entailment_probability=0.9,
            contradiction_probability=0.05,
            neutral_probability=0.05,
        ),
        rationale="high confidence supports",
    )
    pr2 = PassageVerificationResult(
        span_id="s2",
        verdict=VerificationVerdict.CONTRADICTED,
        confidence=0.8,
        probability_distribution=PassageVerificationScore(
            entailment_probability=0.05,
            contradiction_probability=0.8,
            neutral_probability=0.15,
        ),
        rationale="high confidence contradicts",
    )
    pr3 = PassageVerificationResult(
        span_id="s3",
        verdict=VerificationVerdict.INSUFFICIENT,
        confidence=0.7,
        probability_distribution=PassageVerificationScore(
            entailment_probability=0.15,
            contradiction_probability=0.15,
            neutral_probability=0.7,
        ),
        rationale="insufficient",
    )
    return (pr1, pr2, pr3)


def test_evidence_weigher_calculation(
    mock_passage_results: tuple[PassageVerificationResult, ...],
) -> None:
    weigher = DefaultEvidenceWeigher()
    pr = mock_passage_results[0]  # confidence = 0.9
    w = weigher.compute_weight(pr, 0.8)
    assert pytest.approx(w) == 0.72


def test_sort_passage_results(
    mock_passage_results: tuple[PassageVerificationResult, ...],
    test_bundle: EvidenceBundle,
) -> None:
    # Shuffle ordering
    shuffled = (
        mock_passage_results[2],
        mock_passage_results[0],
        mock_passage_results[1],
    )
    sorted_res = sort_passage_results(shuffled, test_bundle.passages)

    # Retrieval index sequence is s1, s2, s3
    assert sorted_res[0].span_id == "s1"
    assert sorted_res[1].span_id == "s2"
    assert sorted_res[2].span_id == "s3"


def test_max_confidence_strategy(
    mock_passage_results: tuple[PassageVerificationResult, ...],
    test_bundle: EvidenceBundle,
) -> None:
    strategy = MaxConfidenceAggregationStrategy()
    definition = NLIVerificationDefinition(
        top_k=3, confidence_thresholds={"SUPPORTED": 0.5, "CONTRADICTED": 0.5}
    )
    claim_input = ClaimVerificationInput(
        claim="Water boils at 100C", bundle=test_bundle, definition=definition
    )

    result = strategy.aggregate(claim_input, mock_passage_results)
    assert result.verdict == VerificationVerdict.SUPPORTED
    assert result.confidence == 0.9
    assert result.aggregation_trace is not None
    assert result.conflict_analysis is not None
    assert result.conflict_analysis.supporting_passages == ("s1",)
    assert result.conflict_analysis.contradicting_passages == ("s2",)


def test_weighted_voting_strategy(
    mock_passage_results: tuple[PassageVerificationResult, ...],
    test_bundle: EvidenceBundle,
) -> None:
    # Weights:
    # s1 (SUPPORTED): weight = 0.8 * 0.9 = 0.72
    # s2 (CONTRADICTED): weight = 0.6 * 0.8 = 0.48
    # s3 (INSUFFICIENT): weight = 0.4 * 0.7 = 0.28
    strategy = WeightedVotingAggregationStrategy()
    definition = NLIVerificationDefinition(
        top_k=3, confidence_thresholds={"SUPPORTED": 0.4, "CONTRADICTED": 0.4}
    )
    claim_input = ClaimVerificationInput(
        claim="Water boils at 100C", bundle=test_bundle, definition=definition
    )

    result = strategy.aggregate(claim_input, mock_passage_results)
    assert result.verdict == VerificationVerdict.SUPPORTED
    # Total weight = 0.72 + 0.48 + 0.28 = 1.48
    # SUPPORTED relative weight = 0.72 / 1.48 = 0.4864
    assert pytest.approx(result.confidence) == (0.72 / 1.48)
    assert result.aggregation_trace is not None
    assert (
        pytest.approx(result.aggregation_trace.intermediate_scores["weight_supported"])
        == 0.72
    )


def test_consensus_strategy_met(
    mock_passage_results: tuple[PassageVerificationResult, ...],
    test_bundle: EvidenceBundle,
) -> None:
    # Counts: 1 supports, 1 contradicts, 1 insufficient (ratio is 1/3 = 0.33)
    # Let's test when threshold is met by providing agreement
    pr1 = mock_passage_results[0]
    pr2 = mock_passage_results[0]
    pr3 = mock_passage_results[0]
    agreed_results = (pr1, pr2, pr3)

    strategy = ConsensusAggregationStrategy(consensus_threshold=0.6)
    definition = NLIVerificationDefinition(top_k=3)
    claim_input = ClaimVerificationInput(
        claim="Water boils at 100C", bundle=test_bundle, definition=definition
    )

    result = strategy.aggregate(claim_input, agreed_results)
    assert result.verdict == VerificationVerdict.SUPPORTED
    assert result.confidence == 0.9


def test_consensus_strategy_not_met(
    mock_passage_results: tuple[PassageVerificationResult, ...],
    test_bundle: EvidenceBundle,
) -> None:
    strategy = ConsensusAggregationStrategy(consensus_threshold=0.6)
    definition = NLIVerificationDefinition(top_k=3)
    claim_input = ClaimVerificationInput(
        claim="Water boils at 100C", bundle=test_bundle, definition=definition
    )

    result = strategy.aggregate(claim_input, mock_passage_results)
    assert result.verdict == VerificationVerdict.INSUFFICIENT
    assert result.confidence == 0.0


def test_contradiction_aware_strategy_resolved(
    mock_passage_results: tuple[PassageVerificationResult, ...],
    test_bundle: EvidenceBundle,
) -> None:
    # Supp: 0.9, Contra: 0.8
    # Severity: 0.8 / 0.9 = 0.888. Imbalance: 0.1
    # Both are >= threshold 0.3, and imbalance < 0.15. So severe contradiction.
    strategy = ContradictionAwareAggregationStrategy(contradiction_threshold=0.3)
    definition = NLIVerificationDefinition(top_k=3)
    claim_input = ClaimVerificationInput(
        claim="Water boils at 100C", bundle=test_bundle, definition=definition
    )

    result = strategy.aggregate(claim_input, mock_passage_results)
    assert result.verdict == VerificationVerdict.INSUFFICIENT
    # Conf = max(0.9, 0.8) * 0.5 = 0.45
    assert result.confidence == 0.45


def test_registry_profile_validations() -> None:
    # Invalid strategy profile definition
    with pytest.raises(ValidationError):
        AggregationProfile(
            profile_id="test",
            strategy_type=AggregationStrategyType.MAX_CONFIDENCE,
            strategy="invalid_strategy_obj",
        )

    # Valid
    p1 = AggregationProfile(
        profile_id="max_conf",
        strategy_type=AggregationStrategyType.MAX_CONFIDENCE,
        strategy=MaxConfidenceAggregationStrategy(),
    )
    p2 = AggregationProfile(
        profile_id="max_conf",  # Duplicate ID
        strategy_type=AggregationStrategyType.MAX_CONFIDENCE,
        strategy=MaxConfidenceAggregationStrategy(),
    )
    with pytest.raises(ValidationError):
        AggregationProfileRegistry(profiles=(p1, p2))
