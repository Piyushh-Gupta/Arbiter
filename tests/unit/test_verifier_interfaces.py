"""Comprehensive unit tests for M2.2 Verifier Interfaces & Immutable Models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.core.bootstrap import DummyNLIModel, build_verification_registry
from src.core.config import Settings
from src.core.exceptions import DuplicateVerificationProfileError
from src.core.retrieval.retrieval_models import (
    EvidenceBundle,
    EvidencePassage,
    RetrievalMetadata,
)
from src.core.verification.aggregation import MaxConfidenceAggregationStrategy
from src.core.verification.implementations import DefaultMetadataProvider, NLIVerifier
from src.core.verification.verification_models import (
    AggregationMetadata,
    ClaimVerificationContext,
    ClaimVerificationInput,
    ExecutionDevice,
    NLIVerificationDefinition,
    PassageVerificationInput,
    PassageVerificationMetadata,
    PassageVerificationResult,
    PassageVerificationScore,
    ProbabilitySchema,
    VerificationDefinition,
    VerificationExecutionMetadata,
    VerificationProfile,
    VerificationProfileRegistry,
    VerificationResult,
    VerificationVerdict,
    VerifierRuntimeMetadata,
)


def test_execution_device_enum() -> None:
    assert ExecutionDevice.CPU == "CPU"
    assert ExecutionDevice.CUDA == "CUDA"
    assert ExecutionDevice.MPS == "MPS"
    assert ExecutionDevice.TPU == "TPU"
    assert ExecutionDevice.OTHER == "OTHER"


def test_probability_schema_immutability() -> None:
    schema = ProbabilitySchema(
        supported_labels=("SUPPORTED", "CONTRADICTED", "INSUFFICIENT"),
        probability_ordering=("SUPPORTED", "CONTRADICTED", "INSUFFICIENT"),
        tolerance=1e-4,
    )
    with pytest.raises(ValidationError):
        schema.tolerance = 1e-5


def test_passage_verification_score_validation() -> None:
    # 1. Valid score
    score = PassageVerificationScore(
        entailment_probability=0.8,
        contradiction_probability=0.1,
        neutral_probability=0.1,
    )
    assert score.entailment_probability == 0.8
    assert score.contradiction_probability == 0.1
    assert score.neutral_probability == 0.1

    # 2. Sum does not equal 1.0 within tolerance
    with pytest.raises(ValidationError):
        PassageVerificationScore(
            entailment_probability=0.5,
            contradiction_probability=0.1,
            neutral_probability=0.1,
        )

    # 3. Contains NaN
    with pytest.raises(ValidationError):
        PassageVerificationScore(
            entailment_probability=float("nan"),
            contradiction_probability=0.5,
            neutral_probability=0.5,
        )

    # 4. Contains Infinity
    with pytest.raises(ValidationError):
        PassageVerificationScore(
            entailment_probability=float("inf"),
            contradiction_probability=-float("inf"),
            neutral_probability=0.5,
        )

    # 5. Out of bounds (e.g. negative or > 1.0)
    with pytest.raises(ValidationError):
        PassageVerificationScore(
            entailment_probability=-0.1,
            contradiction_probability=0.6,
            neutral_probability=0.5,
        )


def test_passage_verification_score_schema_conformance() -> None:
    schema = ProbabilitySchema(
        supported_labels=("SUPPORTED", "CONTRADICTED", "INSUFFICIENT"),
        probability_ordering=("SUPPORTED", "CONTRADICTED", "INSUFFICIENT"),
        tolerance=1e-5,
    )
    # Valid
    score = PassageVerificationScore(
        entailment_probability=0.9,
        contradiction_probability=0.05,
        neutral_probability=0.05,
    )
    assert score.conforms_to_schema(schema) is True
    score.validate_against_schema(schema)

    # Sum within tolerance of 1e-5 (e.g. 1.000004), but not schema tolerance of 1e-9
    # Since 0.900004 + 0.05 + 0.05 = 1.000004
    bad_score = PassageVerificationScore(
        entailment_probability=0.900004,
        contradiction_probability=0.05,
        neutral_probability=0.05,
    )
    # Pydantic validates sum is 1.0 within 1e-5 because our core model validator uses 1e-5.
    # Let's check with tolerance exceeding schemas. E.g. a schema with a very tight tolerance like 1e-9.
    tight_schema = ProbabilitySchema(
        supported_labels=("SUPPORTED", "CONTRADICTED", "INSUFFICIENT"),
        probability_ordering=("SUPPORTED", "CONTRADICTED", "INSUFFICIENT"),
        tolerance=1e-9,
    )
    assert bad_score.conforms_to_schema(tight_schema) is False
    with pytest.raises(ValueError, match="does not conform"):
        bad_score.validate_against_schema(tight_schema)


def test_metadata_strong_typing() -> None:
    # Verify PassageVerificationMetadata properties are typed
    meta = PassageVerificationMetadata(
        model_version="1.2",
        inference_precision="fp16",
        device_used=ExecutionDevice.CUDA,
    )
    assert meta.model_version == "1.2"
    assert meta.device_used == ExecutionDevice.CUDA

    with pytest.raises(ValidationError):
        meta.model_version = "2.0"


def test_verifier_runtime_metadata_timezone_aware() -> None:
    # 1. Valid timezone-aware datetime
    runtime_meta = VerifierRuntimeMetadata(
        model_id="test-nli-1",
        revision="abc1234",
        tokenizer="xlm-roberta",
        framework="pytorch",
        execution_device=ExecutionDevice.CPU,
        inference_precision="fp32",
        execution_timestamp=datetime.now(timezone.utc),
    )
    assert runtime_meta.model_id == "test-nli-1"
    assert runtime_meta.execution_device == ExecutionDevice.CPU
    assert runtime_meta.execution_timestamp.tzinfo is not None

    # 2. Invalid (naive datetime)
    with pytest.raises(ValidationError, match="timezone-aware"):
        VerifierRuntimeMetadata(
            model_id="test-nli-1",
            revision="abc1234",
            tokenizer="xlm-roberta",
            framework="pytorch",
            execution_device=ExecutionDevice.CPU,
            inference_precision="fp32",
            execution_timestamp=datetime.now(),  # Naive
        )


def test_verification_execution_metadata_immutability() -> None:
    exec_meta = VerificationExecutionMetadata(
        request_id="req-123",
        execution_duration=0.45,
        verifier_profile="default",
        aggregation_profile="max_conf",
        configuration_fingerprint="sha256hashstring",
    )
    assert exec_meta.request_id == "req-123"
    with pytest.raises(ValidationError):
        exec_meta.execution_duration = 0.50


def test_passage_verification_input_immutability() -> None:
    passage = EvidencePassage(
        document_id="doc1",
        span_id="span-1",
        text="Sample text",
        score=0.9,
    )
    exec_meta = VerificationExecutionMetadata(
        request_id="req-123",
        execution_duration=0.45,
        verifier_profile="default",
        aggregation_profile="max_conf",
        configuration_fingerprint="sha256hashstring",
    )
    p_input = PassageVerificationInput(
        claim="Claim text",
        passage=passage,
        execution_metadata=exec_meta,
    )
    assert p_input.claim == "Claim text"
    with pytest.raises(ValidationError):
        p_input.claim = "Different"


def test_claim_verification_input_immutability() -> None:
    bundle = EvidenceBundle(
        claim="Claim text",
        passages=(),
        metadata=RetrievalMetadata(strategy_id="test", top_k=5),
    )
    definition = VerificationDefinition()
    c_input = ClaimVerificationInput(
        claim="Claim text",
        bundle=bundle,
        definition=definition,
    )
    assert c_input.claim == "Claim text"
    with pytest.raises(ValidationError):
        c_input.claim = "Different"


def test_claim_verification_context_immutability() -> None:
    score = PassageVerificationScore(
        entailment_probability=0.7,
        contradiction_probability=0.1,
        neutral_probability=0.2,
    )
    pr = PassageVerificationResult(
        span_id="span-1",
        verdict=VerificationVerdict.SUPPORTED,
        confidence=0.7,
        probability_distribution=score,
    )
    agg_meta = AggregationMetadata(
        strategy_id="max_conf",
        thresholds_applied={"SUPPORTED": 0.5},
    )
    exec_meta = VerificationExecutionMetadata(
        request_id="req-123",
        execution_duration=0.45,
        verifier_profile="default",
        aggregation_profile="max_conf",
        configuration_fingerprint="sha256hashstring",
    )
    ctx = ClaimVerificationContext(
        ordered_passage_results=(pr,),
        aggregation_metadata=agg_meta,
        execution_metadata=exec_meta,
    )
    assert ctx.aggregation_metadata.strategy_id == "max_conf"
    with pytest.raises(ValidationError):
        ctx.aggregation_metadata = agg_meta


def test_profile_registry_and_resolution() -> None:
    engine = NLIVerifier(model=DummyNLIModel(), strategy_id="dummy_nli")
    metadata_provider = DefaultMetadataProvider(model_id="nli-default")
    prob_schema = ProbabilitySchema()
    agg_strategy = MaxConfidenceAggregationStrategy()

    definition = NLIVerificationDefinition(
        top_k=5,
        probability_schema=prob_schema,
        aggregation_strategy=agg_strategy,
    )

    profile = VerificationProfile(
        profile_id="test_profile",
        definition=definition,
        verifier=engine,
        metadata_provider=metadata_provider,
    )

    registry = VerificationProfileRegistry(profiles=(profile,))
    assert registry.resolve("test_profile") is profile

    with pytest.raises(DuplicateVerificationProfileError):
        VerificationProfileRegistry(profiles=(profile, profile))


def test_bootstrap_building_and_fail_fast() -> None:
    settings = Settings()
    registry = build_verification_registry(settings)
    profile = registry.resolve("default_verification")
    assert profile.profile_id == "default_verification"
    assert profile.metadata_provider is not None


def test_pipeline_integration_flow() -> None:
    # Verify ClaimVerificationInput -> BaseVerifier -> PassageVerificationResults -> AggregationStrategy -> VerificationResult
    verifier = NLIVerifier(model=DummyNLIModel(), strategy_id="dummy_nli")
    definition = NLIVerificationDefinition(top_k=2)

    passage1 = EvidencePassage(
        document_id="doc1",
        span_id="span-1",
        text="Entailment text.",
        score=0.9,
    )
    bundle = EvidenceBundle(
        claim="Water boils at 100 degrees Celsius",
        passages=(passage1,),
        metadata=RetrievalMetadata(strategy_id="test", top_k=1),
    )

    claim_input = ClaimVerificationInput(
        claim="Water boils at 100 degrees Celsius",
        bundle=bundle,
        definition=definition,
    )

    # 1. BaseVerifier (NLIVerifier) verify_passages
    passage_results = verifier.verify_passages(claim_input.claim, claim_input.bundle)
    assert len(passage_results) == 1
    assert passage_results[0].span_id == "span-1"
    assert passage_results[0].verdict == VerificationVerdict.SUPPORTED

    # 2. Aggregation Strategy
    strategy = MaxConfidenceAggregationStrategy()
    result = strategy.aggregate(
        verification_input=claim_input,
        passage_results=passage_results,
    )

    assert isinstance(result, VerificationResult)
    assert result.verdict == VerificationVerdict.SUPPORTED
    assert result.confidence == 1.0
    assert result.supporting_passages == ("span-1",)


def test_nli_verifier_determinism() -> None:
    verifier = NLIVerifier(model=DummyNLIModel(), strategy_id="dummy_nli")
    definition = NLIVerificationDefinition(top_k=2)

    passage1 = EvidencePassage(
        document_id="doc1",
        span_id="span-1",
        text="Direct support.",
        score=0.9,
    )
    bundle = EvidenceBundle(
        claim="Test",
        passages=(passage1,),
        metadata=RetrievalMetadata(strategy_id="test", top_k=1),
    )

    res1 = verifier.verify("Test", bundle, definition)
    res2 = verifier.verify("Test", bundle, definition)

    assert res1 == res2
    assert res1.verdict == VerificationVerdict.SUPPORTED
    assert res1.confidence == 1.0
