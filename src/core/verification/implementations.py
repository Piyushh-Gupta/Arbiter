"""Concrete implementations for the Verification subsystem."""

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from src.core.exceptions import (
    VerificationConfigurationError,
    VerificationExecutionError,
)
from src.core.retrieval.retrieval_models import EvidenceBundle
from src.core.verification.verification_models import (
    NLIVerificationDefinition,
    VerificationDefinition,
    VerificationLabel,
    VerificationMetadata,
    VerificationResult,
)


@runtime_checkable
class NLIModel(Protocol):
    """Stateless protocol for NLI model backend."""

    @property
    def label_map(self) -> Mapping[int, VerificationLabel]:
        """Translates the model's internal output indices to canonical VerificationLabels."""
        ...

    def predict(
        self, claim: str, passages: Sequence[str]
    ) -> list[tuple[float, float, float]]:
        """
        Scores a claim against a batch of passages.

        Receives:
        - claim: The normalized textual assertion.
        - passages: An ordered sequence of evidence passage texts.

        Returns:
        - list[tuple[float, float, float]]: One probability triplet per passage.
          Each triplet is (p_supports, p_refutes, p_nei), where all three
          sum to approximately 1.0. The order matches the input passages exactly.

        Must not perform filesystem or network access during inference.
        Must not cache internal state.
        """
        ...


class NLIVerifier:
    """
    NLI-based verification strategy using per-label max pooling evidence aggregation.
    """

    def __init__(self, model: NLIModel, strategy_id: str) -> None:
        self.model = model
        self._strategy_id = strategy_id

    def validate_compatibility(self, definition: VerificationDefinition) -> None:
        if not isinstance(definition, NLIVerificationDefinition):
            raise VerificationConfigurationError(
                f"NLIVerifier requires NLIVerificationDefinition, got {type(definition).__name__}"
            )

        # Validate that model.label_map contains all three labels exactly once.
        label_values = list(self.model.label_map.values())
        if len(label_values) != 3 or set(label_values) != {
            VerificationLabel.SUPPORTS,
            VerificationLabel.REFUTES,
            VerificationLabel.NOT_ENOUGH_INFO,
        }:
            raise VerificationConfigurationError(
                "NLIModel.label_map must contain all three VerificationLabel values exactly once."
            )

    def verify(
        self,
        claim: str,
        bundle: EvidenceBundle,
        definition: VerificationDefinition,
    ) -> VerificationResult:
        if not isinstance(definition, NLIVerificationDefinition):
            raise VerificationConfigurationError(
                f"NLIVerifier requires NLIVerificationDefinition, got {type(definition).__name__}"
            )

        passages_to_score = bundle.passages[: definition.top_k]

        if not passages_to_score:
            return VerificationResult(
                label=VerificationLabel.NOT_ENOUGH_INFO,
                confidence=None,
                evidence_bundle=bundle,
                metadata=VerificationMetadata(strategy_id=self._strategy_id),
            )

        passage_texts = [p.text for p in passages_to_score]

        try:
            raw_predictions = self.model.predict(claim, passage_texts)
        except Exception as e:
            raise VerificationExecutionError(f"NLI model execution failed: {e}") from e

        if len(raw_predictions) != len(passages_to_score):
            raise VerificationExecutionError(
                f"NLI model returned {len(raw_predictions)} predictions for {len(passages_to_score)} passages."
            )

        # Build inverse map: Label -> Index
        label_to_idx = {v: k for k, v in self.model.label_map.items()}

        idx_supports = label_to_idx[VerificationLabel.SUPPORTS]
        idx_refutes = label_to_idx[VerificationLabel.REFUTES]
        idx_nei = label_to_idx[VerificationLabel.NOT_ENOUGH_INFO]

        max_supports = -1.0
        max_refutes = -1.0
        max_nei = -1.0

        for triplet in raw_predictions:
            if triplet[idx_supports] > max_supports:
                max_supports = triplet[idx_supports]
            if triplet[idx_refutes] > max_refutes:
                max_refutes = triplet[idx_refutes]
            if triplet[idx_nei] > max_nei:
                max_nei = triplet[idx_nei]

        # Tie-breaking logic: SUPPORTS > REFUTES > NOT_ENOUGH_INFO
        if max_supports >= max_refutes and max_supports >= max_nei:
            winning_label = VerificationLabel.SUPPORTS
            winning_confidence = max_supports
        elif max_refutes > max_supports and max_refutes >= max_nei:
            winning_label = VerificationLabel.REFUTES
            winning_confidence = max_refutes
        else:
            winning_label = VerificationLabel.NOT_ENOUGH_INFO
            winning_confidence = max_nei

        # Round confidence slightly to avoid floating point issues triggering pydantic validation
        winning_confidence = max(0.0, min(1.0, float(winning_confidence)))

        return VerificationResult(
            label=winning_label,
            confidence=winning_confidence,
            evidence_bundle=bundle,
            metadata=VerificationMetadata(strategy_id=self._strategy_id),
        )
