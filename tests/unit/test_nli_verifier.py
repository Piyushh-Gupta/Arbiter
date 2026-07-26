"""Unit tests for the M9.2 NLI Verifier."""

from collections.abc import Mapping, Sequence

import pytest
from pydantic import ValidationError

from src.core.exceptions import (
    VerificationConfigurationError,
    VerificationExecutionError,
)
from src.core.retrieval.retrieval_models import (
    EvidenceBundle,
    EvidencePassage,
    RetrievalMetadata,
)
from src.core.verification.implementations import NLIModel, NLIVerifier
from src.core.verification.verification_models import (
    NLIVerificationDefinition,
    VerificationDefinition,
    VerificationLabel,
)
from src.core.verification.verifier import ClaimVerifier


class MockNLIModel:
    def __init__(
        self,
        label_map: Mapping[int, VerificationLabel] | None = None,
        fail_predict: bool = False,
        wrong_length: bool = False,
        predictions: list[tuple[float, float, float]] | None = None,
    ):
        if label_map is None:
            self._label_map: Mapping[int, VerificationLabel] = {
                0: VerificationLabel.SUPPORTS,
                1: VerificationLabel.REFUTES,
                2: VerificationLabel.NOT_ENOUGH_INFO,
            }
        else:
            self._label_map = label_map

        self.fail_predict = fail_predict
        self.wrong_length = wrong_length
        self.predictions = predictions
        self.called_with_claim: str | None = None
        self.called_with_passages: Sequence[str] | None = None

    @property
    def label_map(self) -> Mapping[int, VerificationLabel]:
        return self._label_map

    def predict(
        self, claim: str, passages: Sequence[str]
    ) -> list[tuple[float, float, float]]:
        if self.fail_predict:
            raise RuntimeError("Model inference failed")

        self.called_with_claim = claim
        self.called_with_passages = passages

        if self.wrong_length:
            return [(0.1, 0.2, 0.7)] * (len(passages) + 1)

        if self.predictions is not None:
            return self.predictions

        # Default mock: all passages return NOT_ENOUGH_INFO (idx 2)
        return [(0.1, 0.2, 0.7)] * len(passages)


@pytest.fixture
def dummy_bundle() -> EvidenceBundle:
    passages = (
        EvidencePassage(
            document_id="d1", span_id="s1", text="p1", score=0.9, metadata={}
        ),
        EvidencePassage(
            document_id="d2", span_id="s2", text="p2", score=0.8, metadata={}
        ),
        EvidencePassage(
            document_id="d3", span_id="s3", text="p3", score=0.7, metadata={}
        ),
    )
    return EvidenceBundle(
        claim="The claim",
        passages=passages,
        metadata=RetrievalMetadata(strategy_id="test", top_k=3),
    )


def test_nli_model_protocol_compliance() -> None:
    assert isinstance(MockNLIModel(), NLIModel)


def test_nli_definition_immutable() -> None:
    definition = NLIVerificationDefinition(top_k=5)
    with pytest.raises(ValidationError):
        definition.top_k = 10

    with pytest.raises(ValidationError):
        NLIVerificationDefinition(top_k=0)


def test_compatibility_validation() -> None:
    model = MockNLIModel()
    verifier = NLIVerifier(model, "test_nli")

    # Valid
    verifier.validate_compatibility(NLIVerificationDefinition(top_k=3))

    # Wrong type
    with pytest.raises(
        VerificationConfigurationError, match="requires NLIVerificationDefinition"
    ):
        verifier.validate_compatibility(VerificationDefinition())

    # Incomplete label_map
    bad_model = MockNLIModel(
        label_map={0: VerificationLabel.SUPPORTS, 1: VerificationLabel.REFUTES}
    )
    bad_verifier = NLIVerifier(bad_model, "bad")
    with pytest.raises(
        VerificationConfigurationError,
        match="must contain all three VerificationLabel values exactly once",
    ):
        bad_verifier.validate_compatibility(NLIVerificationDefinition(top_k=3))


def test_empty_bundle_fast_path() -> None:
    model = MockNLIModel()
    verifier = NLIVerifier(model, "test")
    empty_bundle = EvidenceBundle(
        claim="Test claim",
        passages=(),
        metadata=RetrievalMetadata(strategy_id="test", top_k=3),
    )
    definition = NLIVerificationDefinition(top_k=3)

    result = verifier.verify("Test claim", empty_bundle, definition)

    assert result.label == VerificationLabel.NOT_ENOUGH_INFO
    assert result.confidence is None
    assert result.evidence_bundle is empty_bundle
    assert model.called_with_passages is None  # predict not called


def test_top_k_slicing(dummy_bundle: EvidenceBundle) -> None:
    model = MockNLIModel()
    verifier = NLIVerifier(model, "test")

    # Bundle has 3 passages, definition requests top_k=2
    definition = NLIVerificationDefinition(top_k=2)
    verifier.verify("Test claim", dummy_bundle, definition)

    assert model.called_with_passages == ["p1", "p2"]


def test_max_pooling_aggregation_and_label_mapping(
    dummy_bundle: EvidenceBundle,
) -> None:
    # Let model mapping be: 0: REFUTES, 1: NEI, 2: SUPPORTS
    label_map = {
        0: VerificationLabel.REFUTES,
        1: VerificationLabel.NOT_ENOUGH_INFO,
        2: VerificationLabel.SUPPORTS,
    }
    # Passages:
    # 1: Refutes (0.9), NEI (0.1), Supports (0.0)
    # 2: Refutes (0.1), NEI (0.1), Supports (0.8)
    # 3: Refutes (0.2), NEI (0.7), Supports (0.1)
    predictions = [
        (0.9, 0.1, 0.0),
        (0.1, 0.1, 0.8),
        (0.2, 0.7, 0.1),
    ]

    model = MockNLIModel(label_map=label_map, predictions=predictions)
    verifier = NLIVerifier(model, "test")

    result = verifier.verify(
        "Test claim", dummy_bundle, NLIVerificationDefinition(top_k=3)
    )

    # Max pooled: REFUTES=0.9, NEI=0.7, SUPPORTS=0.8
    # REFUTES should win
    assert result.label == VerificationLabel.REFUTES
    assert result.confidence == 0.9
    assert result.evidence_bundle is dummy_bundle


def test_tie_breaking(dummy_bundle: EvidenceBundle) -> None:
    # Default map: 0: SUPPORTS, 1: REFUTES, 2: NEI
    # Tie between SUPPORTS and REFUTES
    predictions = [
        (0.8, 0.8, 0.1),
    ]
    model = MockNLIModel(predictions=predictions)
    verifier = NLIVerifier(model, "test")

    result = verifier.verify(
        "Test claim", dummy_bundle, NLIVerificationDefinition(top_k=1)
    )

    # SUPPORTS > REFUTES > NEI
    assert result.label == VerificationLabel.SUPPORTS
    assert result.confidence == 0.8


def test_exception_wrapping(dummy_bundle: EvidenceBundle) -> None:
    model = MockNLIModel(fail_predict=True)
    verifier = NLIVerifier(model, "test")

    with pytest.raises(
        VerificationExecutionError,
        match="NLI model execution failed: Model inference failed",
    ):
        verifier.verify("Test claim", dummy_bundle, NLIVerificationDefinition(top_k=3))


def test_output_length_mismatch(dummy_bundle: EvidenceBundle) -> None:
    model = MockNLIModel(wrong_length=True)
    verifier = NLIVerifier(model, "test")

    with pytest.raises(
        VerificationExecutionError,
        match="NLI model returned 4 predictions for 3 passages.",
    ):
        verifier.verify("Test claim", dummy_bundle, NLIVerificationDefinition(top_k=3))


def test_execution_equivalence(dummy_bundle: EvidenceBundle) -> None:
    model = MockNLIModel()
    verifier = NLIVerifier(model, "test")
    orchestrator = ClaimVerifier()
    definition = NLIVerificationDefinition(top_k=3)

    # Via orchestrator
    result_orch = orchestrator.verify("Test claim", dummy_bundle, definition, verifier)

    # Direct
    result_direct = verifier.verify("Test claim", dummy_bundle, definition)

    assert result_orch.label == result_direct.label
    assert result_orch.confidence == result_direct.confidence
    assert result_orch.evidence_bundle is dummy_bundle
