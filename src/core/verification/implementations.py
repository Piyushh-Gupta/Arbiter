"""Concrete implementations for the Verification subsystem."""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
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
    ClaimVerificationInput,
    ExecutionDevice,
    NLIVerificationDefinition,
    PassageVerificationMetadata,
    PassageVerificationResult,
    PassageVerificationScore,
    VerificationDefinition,
    VerificationLabel,
    VerificationMetadata,
    VerificationResult,
    VerificationVerdict,
    VerifiedPassage,
    VerifierRuntimeMetadata,
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


class DefaultMetadataProvider:
    """Default implementation of metadata provider."""

    def __init__(self, model_id: str = "nli-default"):
        self.model_id = model_id

    def get_runtime_metadata(self) -> VerifierRuntimeMetadata:
        return VerifierRuntimeMetadata(
            model_id=self.model_id,
            revision="1.0",
            tokenizer="default",
            framework="pytorch",
            execution_device=ExecutionDevice.CPU,
            inference_precision="fp32",
            execution_timestamp=datetime.now(timezone.utc),
        )


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
            p_supports_raw = float(max(0.0, min(1.0, triplet[idx_supports])))
            p_refutes_raw = float(max(0.0, min(1.0, triplet[idx_refutes])))
            p_nei_raw = float(max(0.0, min(1.0, triplet[idx_nei])))

            # Determine verdict and raw confidence
            if p_supports_raw >= p_refutes_raw and p_supports_raw >= p_nei_raw:
                verdict = VerificationVerdict.SUPPORTED
                conf = p_supports_raw
            elif p_refutes_raw > p_supports_raw and p_refutes_raw >= p_nei_raw:
                verdict = VerificationVerdict.CONTRADICTED
                conf = p_refutes_raw
            else:
                verdict = VerificationVerdict.INSUFFICIENT
                conf = p_nei_raw

            # Calculate normalized probabilities for the score distribution
            total = p_supports_raw + p_refutes_raw + p_nei_raw
            if total > 0.0:
                p_supports = p_supports_raw / total
                p_refutes = p_refutes_raw / total
                p_nei = p_nei_raw / total
            else:
                p_supports, p_refutes, p_nei = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0

            score = PassageVerificationScore(
                entailment_probability=p_supports,
                contradiction_probability=p_refutes,
                neutral_probability=p_nei,
            )

            passage_results.append(
                PassageVerificationResult(
                    span_id=passage.span_id,
                    verdict=verdict,
                    confidence=max(0.0, min(1.0, conf)),
                    probability_distribution=score,
                    rationale="NLI prediction",
                    latency=0.01,
                    metadata=PassageVerificationMetadata(
                        model_version="1.0",
                        inference_precision="fp32",
                        device_used=ExecutionDevice.CPU,
                    ),
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

        # Validate each score against the active schema in the definition
        for pr in passage_results:
            pr.probability_distribution.validate_against_schema(
                definition.probability_schema
            )

        if isinstance(definition.aggregation_strategy, BaseAggregationStrategy):
            strategy = definition.aggregation_strategy
        else:
            strategy = MaxConfidenceAggregationStrategy()

        metadata_provider = DefaultMetadataProvider(model_id=definition.verifier_model)
        runtime_metadata = metadata_provider.get_runtime_metadata()

        claim_input = ClaimVerificationInput(
            claim=claim,
            bundle=sub_bundle,
            definition=definition,
        )

        res = strategy.aggregate(
            verification_input=claim_input,
            passage_results=passage_results,
            runtime_metadata=runtime_metadata,
        )

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
