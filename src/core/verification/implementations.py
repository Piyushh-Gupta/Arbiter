"""Concrete implementations for the Verification subsystem."""

from datetime import UTC, datetime

from src.core.exceptions import (
    VerificationConfigurationError,
    VerificationExecutionError,
)
from src.core.retrieval.retrieval_models import EvidenceBundle
from src.core.verification.aggregation import (
    BaseAggregationStrategy,
    MaxConfidenceAggregationStrategy,
)
from src.core.verification.base import BaseNLIModel, BaseVerifier
from src.core.verification.verification_models import (
    LABEL_TO_VERDICT,
    VERDICT_TO_LABEL,
    ClaimVerificationInput,
    ExecutionDevice,
    NLIModelDefinition,
    NLIVerificationDefinition,
    PassageVerificationInput,
    PassageVerificationMetadata,
    PassageVerificationResult,
    VerificationDefinition,
    VerificationExecutionMetadata,
    VerificationLabel,
    VerificationMetadata,
    VerificationResult,
    VerificationVerdict,
    VerifiedPassage,
    VerifierRuntimeMetadata,
)


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
            execution_timestamp=datetime.now(UTC),
        )


class NLIVerifier(BaseVerifier):
    """
    NLI-based verification strategy implementing passage verification and aggregation.
    """

    def __init__(self, model: BaseNLIModel, strategy_id: str) -> None:
        self.model = model
        self._strategy_id = strategy_id

    def validate_compatibility(self, definition: VerificationDefinition) -> None:
        if not isinstance(definition, NLIVerificationDefinition):
            raise VerificationConfigurationError(
                f"NLIVerifier requires NLIVerificationDefinition, got {type(definition).__name__}"
            )

        if hasattr(self.model, "label_map"):
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

        # Construct default execution metadata for inputs wrapping
        exec_meta = VerificationExecutionMetadata(
            request_id="default-req",
            execution_duration=0.0,
            verifier_profile=self._strategy_id,
            aggregation_profile="max_confidence",
            configuration_fingerprint="default_nli_fingerprint",
        )

        inputs = tuple(
            PassageVerificationInput(
                claim=claim,
                passage=p,
                execution_metadata=exec_meta,
            )
            for p in passages_to_score
        )

        try:
            scores = self.model.predict(inputs)
        except Exception as e:
            raise VerificationExecutionError(f"NLI model execution failed: {e}") from e

        if len(scores) != len(passages_to_score):
            raise VerificationExecutionError(
                f"NLI model returned {len(scores)} predictions for {len(passages_to_score)} passages."
            )

        passage_results: list[PassageVerificationResult] = []
        for passage, score in zip(passages_to_score, scores):
            p_supports = score.entailment_probability
            p_refutes = score.contradiction_probability
            p_nei = score.neutral_probability

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

        for pr in passage_results:
            pr.probability_distribution.validate_against_schema(
                definition.probability_schema
            )

        if isinstance(definition.aggregation_strategy, BaseAggregationStrategy):
            strategy = definition.aggregation_strategy
        else:
            strategy = MaxConfidenceAggregationStrategy()

        # Check if model has a configuration to populate VerifierRuntimeMetadata dynamically
        if hasattr(self.model, "config") and isinstance(
            self.model.config, NLIModelDefinition
        ):
            runtime_metadata = VerifierRuntimeMetadata(
                model_id=self.model.config.model_id,
                revision="1.0",
                tokenizer=self.model.config.tokenizer_id,
                framework="pytorch",
                execution_device=self.model.config.execution_device,
                inference_precision=self.model.config.inference_precision,
                execution_timestamp=datetime.now(UTC),
            )
        else:
            metadata_provider = DefaultMetadataProvider(
                model_id=definition.verifier_model
            )
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
