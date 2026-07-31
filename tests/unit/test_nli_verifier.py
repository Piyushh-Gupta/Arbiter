"""Unit tests for the M2.3 NLI Verifier and TransformerNLIModel."""

from collections.abc import Mapping, Sequence
from typing import Any

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
from src.core.verification.base import BaseNLIModel
from src.core.verification.implementations import NLIVerifier
from src.core.verification.verification_models import (
    NLILabelSchema,
    NLIVerificationDefinition,
    PassageVerificationInput,
    PassageVerificationScore,
    VerificationDefinition,
    VerificationLabel,
    VerificationVerdict,
)
from src.core.verification.verifier import ClaimVerifier


class MockNLIModel(BaseNLIModel):
    def __init__(
        self,
        label_map: Mapping[int, Any] | None = None,
        fail_predict: bool = False,
        wrong_length: bool = False,
        predictions: list[tuple[float, float, float]] | None = None,
    ):
        if label_map is None:
            self._label_map = {
                0: VerificationVerdict.SUPPORTED,
                1: VerificationVerdict.CONTRADICTED,
                2: VerificationVerdict.INSUFFICIENT,
            }
        else:
            self._label_map = {
                k: (
                    VerificationVerdict.SUPPORTED
                    if v
                    in (
                        "SUPPORTED",
                        VerificationVerdict.SUPPORTED,
                        VerificationLabel.SUPPORTS,
                        "SUPPORTS",
                    )
                    else VerificationVerdict.CONTRADICTED
                    if v
                    in (
                        "CONTRADICTED",
                        VerificationVerdict.CONTRADICTED,
                        VerificationLabel.REFUTES,
                        "REFUTES",
                    )
                    else VerificationVerdict.INSUFFICIENT
                )
                for k, v in label_map.items()
            }

        self.fail_predict = fail_predict
        self.wrong_length = wrong_length
        self.predictions = predictions
        self.called_with_claim: str | None = None
        self.called_with_passages: Sequence[str] | None = None

    @property
    def label_map(self) -> Mapping[int, Any]:
        return self._label_map

    def predict(
        self, batch: tuple[PassageVerificationInput, ...]
    ) -> tuple[PassageVerificationScore, ...]:
        if self.fail_predict:
            raise RuntimeError("Model inference failed")

        if not batch:
            return ()

        self.called_with_claim = batch[0].claim
        self.called_with_passages = [inp.passage.text for inp in batch]

        num_passages = len(batch)
        if self.wrong_length:
            num_passages += 1

        raw_preds = self.predictions
        if raw_preds is None:
            raw_preds = [(0.15, 0.15, 0.7)] * num_passages

        if len(raw_preds) < num_passages:
            raw_preds = raw_preds * num_passages

        scores = []
        for triplet in raw_preds[:num_passages]:
            entailment = 0.0
            contradiction = 0.0
            neutral = 0.0

            for idx, val in enumerate(triplet):
                v_type = self._label_map.get(idx, VerificationVerdict.INSUFFICIENT)
                if v_type == VerificationVerdict.SUPPORTED:
                    entailment = val
                elif v_type == VerificationVerdict.CONTRADICTED:
                    contradiction = val
                else:
                    neutral = val

            tot = entailment + contradiction + neutral
            if tot > 0.0:
                entailment /= tot
                contradiction /= tot
                neutral /= tot
            else:
                entailment, contradiction, neutral = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0

            scores.append(
                PassageVerificationScore(
                    entailment_probability=entailment,
                    contradiction_probability=contradiction,
                    neutral_probability=neutral,
                )
            )

        return tuple(scores)


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
    assert isinstance(MockNLIModel(), BaseNLIModel)


def test_nli_definition_immutable() -> None:
    definition = NLIVerificationDefinition(top_k=5)
    with pytest.raises(ValidationError):
        definition.top_k = 10


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
    assert result.verified_passages is None
    assert model.called_with_passages is None


def test_top_k_slicing(dummy_bundle: EvidenceBundle) -> None:
    model = MockNLIModel()
    verifier = NLIVerifier(model, "test")

    definition = NLIVerificationDefinition(top_k=2)
    verifier.verify("Test claim", dummy_bundle, definition)

    assert model.called_with_passages == ["p1", "p2"]


def test_max_pooling_aggregation_and_label_mapping(
    dummy_bundle: EvidenceBundle,
) -> None:
    label_map = {
        0: VerificationLabel.REFUTES,
        1: VerificationLabel.NOT_ENOUGH_INFO,
        2: VerificationLabel.SUPPORTS,
    }
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

    assert result.label == VerificationLabel.REFUTES
    # Since confidence is raw winning score
    assert result.confidence == 0.9
    assert result.evidence_bundle is dummy_bundle

    assert result.verified_passages is not None
    assert len(result.verified_passages) == 3

    vp1 = result.verified_passages[0]
    assert vp1.label == VerificationLabel.REFUTES
    assert vp1.refutes_score == 0.9

    vp2 = result.verified_passages[1]
    assert vp2.label == VerificationLabel.SUPPORTS
    assert vp2.supports_score == 0.8

    vp3 = result.verified_passages[2]
    assert vp3.label == VerificationLabel.NOT_ENOUGH_INFO
    assert vp3.not_enough_info_score == 0.7


def test_tie_breaking(dummy_bundle: EvidenceBundle) -> None:
    # Tie between SUPPORTS and REFUTES (index 0 and 1 here)
    # Using values that sum to 1.0 and meet the default 0.5 confidence threshold
    predictions = [
        (0.5, 0.5, 0.0),
    ]
    model = MockNLIModel(predictions=predictions)
    verifier = NLIVerifier(model, "test")

    result = verifier.verify(
        "Test claim", dummy_bundle, NLIVerificationDefinition(top_k=1)
    )

    assert result.label == VerificationLabel.SUPPORTS
    assert result.confidence == 0.5


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

    result_orch = orchestrator.verify("Test claim", dummy_bundle, definition, verifier)
    result_direct = verifier.verify("Test claim", dummy_bundle, definition)

    assert result_orch.label == result_direct.label
    assert result_orch.confidence == result_direct.confidence
    assert result_orch.evidence_bundle is dummy_bundle


def test_nli_label_schema_validation() -> None:
    # Invalid: missing contradiction mapping
    with pytest.raises(
        ValueError,
        match="verdict_mapping must map to all canonical VerificationVerdict values",
    ):
        NLILabelSchema(
            verdict_mapping={
                "SUPPORTED": VerificationVerdict.SUPPORTED,
                "INSUFFICIENT": VerificationVerdict.INSUFFICIENT,
            }
        )
