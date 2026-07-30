"""Concrete implementations for the Verification subsystem."""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from src.core.exceptions import (
    VerificationConfigurationError,
    VerificationExecutionError,
)
from src.core.retrieval.retrieval_models import EvidenceBundle
from src.core.verification.aggregation import (
    BaseAggregationStrategy,
    MaxConfidenceAggregationStrategy,
)
from src.core.verification.verification_models import (
    LABEL_TO_VERDICT,
    VERDICT_TO_LABEL,
    NLIVerificationDefinition,
    PassageVerificationResult,
    VerificationDefinition,
    VerificationLabel,
    VerificationMetadata,
    VerificationModelMetadata,
    VerificationResult,
    VerificationVerdict,
    VerifiedPassage,
)


@runtime_checkable
class NLIModel(Protocol):
    """Stateless protocol for NLI model backend."""

    @property
    def label_map(self) -> Mapping[int, Any]:
        """Translates the model's internal output indices to canonical VerificationVerdict or VerificationLabel values."""
        ...

    def predict(
        self, claim: str, passages: Sequence[str]
    ) -> list[tuple[float, float, float]]:
        """
        Scores a claim against a batch of passages.

        Returns list[tuple[float, float, float]]: (p_supports, p_refutes, p_nei).
        """
        ...


class NLIVerifier:
    """
    NLI-based verification strategy implementing passage verification and aggregation.
    """

    def __init__(self, model: NLIModel, strategy_id: str) -> None:
        self.model = model
        self._strategy_id = strategy_id

    def validate_compatibility(self, definition: VerificationDefinition) -> None:
        if not isinstance(definition, NLIVerificationDefinition):
            raise VerificationConfigurationError(
                f"NLIVerifier requires NLIVerificationDefinition, got {type(definition).__name__}"
            )

        label_values = [
            LABEL_TO_VERDICT.get(v, v) for v in self.model.label_map.values()
        ]
        valid_set = {
            VerificationVerdict.SUPPORTED,
            VerificationVerdict.CONTRADICTED,
            VerificationVerdict.INSUFFICIENT,
        }
        if len(label_values) != 3 or set(label_values) != valid_set:
            raise VerificationConfigurationError(
                "NLIModel.label_map must contain all three VerificationLabel values exactly once."
            )

    def verify_passages(
        self,
        claim: str,
        bundle: EvidenceBundle,
    ) -> tuple[PassageVerificationResult, ...]:
        passages_to_score = bundle.passages
        if not passages_to_score:
            return ()

        passage_texts = [p.text for p in passages_to_score]
        try:
            raw_predictions = self.model.predict(claim, passage_texts)
        except Exception as e:
            raise VerificationExecutionError(f"NLI model execution failed: {e}") from e

        if len(raw_predictions) != len(passages_to_score):
            raise VerificationExecutionError(
                f"NLI model returned {len(raw_predictions)} predictions for {len(passages_to_score)} passages."
            )

        label_to_idx = {
            LABEL_TO_VERDICT.get(v, v): k for k, v in self.model.label_map.items()
        }
        idx_supports = label_to_idx[VerificationVerdict.SUPPORTED]
        idx_refutes = label_to_idx[VerificationVerdict.CONTRADICTED]
        idx_nei = label_to_idx[VerificationVerdict.INSUFFICIENT]

        passage_results: list[PassageVerificationResult] = []
        for passage, triplet in zip(passages_to_score, raw_predictions):
            p_supports = float(max(0.0, min(1.0, triplet[idx_supports])))
            p_refutes = float(max(0.0, min(1.0, triplet[idx_refutes])))
            p_nei = float(max(0.0, min(1.0, triplet[idx_nei])))

            if p_supports >= p_refutes and p_supports >= p_nei:
                verdict = VerificationVerdict.SUPPORTED
                conf = p_supports
            elif p_refutes > p_supports and p_refutes >= p_nei:
                verdict = VerificationVerdict.CONTRADICTED
                conf = p_refutes
            else:
                verdict = VerificationVerdict.INSUFFICIENT
                conf = p_nei

            passage_results.append(
                PassageVerificationResult(
                    span_id=passage.span_id,
                    verdict=verdict,
                    confidence=max(0.0, min(1.0, conf)),
                    raw_scores={
                        "SUPPORTED": p_supports,
                        "CONTRADICTED": p_refutes,
                        "INSUFFICIENT": p_nei,
                    },
                )
            )

        return tuple(passage_results)

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
                verdict=VerificationVerdict.INSUFFICIENT,
                confidence=None,
                evidence_bundle=bundle,
                verified_passages=None,
                metadata=VerificationMetadata(strategy_id=self._strategy_id),
            )

        sub_bundle = EvidenceBundle(
            claim=bundle.claim,
            passages=passages_to_score,
            metadata=bundle.metadata,
        )

        passage_results = self.verify_passages(claim, sub_bundle)

        if isinstance(definition.aggregation_strategy, BaseAggregationStrategy):
            strategy = definition.aggregation_strategy
        else:
            strategy = MaxConfidenceAggregationStrategy()

        metadata = VerificationModelMetadata(
            model_identifier=definition.verifier_model,
            revision="1.0",
            tokenizer="default",
            execution_device="cpu",
            framework_version="1.0.0",
        )

        res = strategy.aggregate(passage_results, definition, model_metadata=metadata)

        legacy_verified: list[VerifiedPassage] = []
        for p, pr in zip(passages_to_score, passage_results):
            raw = pr.raw_scores or {}
            legacy_verified.append(
                VerifiedPassage(
                    passage=p,
                    label=VERDICT_TO_LABEL.get(
                        pr.verdict, VerificationLabel.NOT_ENOUGH_INFO
                    ),
                    supports_score=raw.get("SUPPORTED", pr.confidence),
                    refutes_score=raw.get("CONTRADICTED", 0.0),
                    not_enough_info_score=raw.get("INSUFFICIENT", 0.0),
                )
            )

        return VerificationResult(
            verdict=res.verdict,
            confidence=res.confidence,
            supporting_passages=res.supporting_passages,
            contradicting_passages=res.contradicting_passages,
            evidence_attribution=res.evidence_attribution,
            explanation=res.explanation,
            uncertainty=res.uncertainty,
            model_metadata=res.model_metadata,
            evidence_bundle=bundle,
            verified_passages=tuple(legacy_verified),
            metadata=VerificationMetadata(strategy_id=self._strategy_id),
        )
