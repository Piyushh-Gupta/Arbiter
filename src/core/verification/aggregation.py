"""Aggregation strategy abstractions and concrete implementations for verification results."""

from typing import Protocol, runtime_checkable

from src.core.verification.verification_models import (
    PassageVerificationResult,
    VerificationDefinition,
    VerificationExplanation,
    VerificationModelMetadata,
    VerificationResult,
    VerificationVerdict,
)


@runtime_checkable
class BaseAggregationStrategy(Protocol):
    """Protocol for aggregating passage-level verification outcomes into claim-level verification results."""

    def aggregate(
        self,
        passage_results: tuple[PassageVerificationResult, ...],
        definition: VerificationDefinition,
        model_metadata: VerificationModelMetadata | None = None,
    ) -> VerificationResult:
        """
        Aggregates passage verification results deterministically.

        Receives:
        - passage_results: Tuple of PassageVerificationResult objects.
        - definition: VerificationDefinition parameters.
        - model_metadata: Optional model execution metadata.

        Returns:
        - VerificationResult: Immutable final claim-level verdict.
        """
        ...


class MaxConfidenceAggregationStrategy(BaseAggregationStrategy):
    """
    Max-confidence aggregation strategy.
    Determines claim verdict based on maximum confidence across supporting vs contradicting passages,
    thresholded by definition rules.
    """

    def aggregate(
        self,
        passage_results: tuple[PassageVerificationResult, ...],
        definition: VerificationDefinition,
        model_metadata: VerificationModelMetadata | None = None,
    ) -> VerificationResult:
        if model_metadata is None:
            model_metadata = VerificationModelMetadata(
                model_identifier=definition.verifier_model
            )

        if not passage_results:
            explanation = VerificationExplanation(
                reasoning="No evidence passages were evaluated.",
                confidence_explanation="Confidence is 0.0 due to absence of evaluated passages.",
                uncertainty_explanation="Uncertainty is 1.0 due to lack of evidence.",
            )
            return VerificationResult(
                verdict=VerificationVerdict.INSUFFICIENT,
                confidence=0.0,
                supporting_passages=(),
                contradicting_passages=(),
                evidence_attribution={},
                explanation=explanation,
                uncertainty=1.0,
                model_metadata=model_metadata,
            )

        supporting: list[str] = []
        contradicting: list[str] = []
        attribution: dict[str, float] = {}

        max_supp_conf = -1.0
        max_contra_conf = -1.0
        max_insuff_conf = -1.0

        for p_res in passage_results:
            attribution[p_res.span_id] = float(p_res.confidence)

            if p_res.verdict == VerificationVerdict.SUPPORTED:
                supporting.append(p_res.span_id)
                if p_res.confidence > max_supp_conf:
                    max_supp_conf = p_res.confidence
            elif p_res.verdict == VerificationVerdict.CONTRADICTED:
                contradicting.append(p_res.span_id)
                if p_res.confidence > max_contra_conf:
                    max_contra_conf = p_res.confidence
            else:
                if p_res.confidence > max_insuff_conf:
                    max_insuff_conf = p_res.confidence

        supp_thresh = definition.confidence_thresholds.get("SUPPORTED", 0.5)
        contra_thresh = definition.confidence_thresholds.get("CONTRADICTED", 0.5)

        valid_supp = max_supp_conf >= supp_thresh
        valid_contra = max_contra_conf >= contra_thresh

        if valid_supp and (not valid_contra or max_supp_conf >= max_contra_conf):
            final_verdict = VerificationVerdict.SUPPORTED
            final_conf = max(0.0, min(1.0, max_supp_conf))
        elif valid_contra and (not valid_supp or max_contra_conf > max_supp_conf):
            final_verdict = VerificationVerdict.CONTRADICTED
            final_conf = max(0.0, min(1.0, max_contra_conf))
        else:
            final_verdict = VerificationVerdict.INSUFFICIENT
            final_conf = max(
                0.0,
                min(
                    1.0,
                    max(max_supp_conf, max_contra_conf, max_insuff_conf, 0.0),
                ),
            )

        reasoning_text = (
            f"Evaluated {len(passage_results)} passage(s). Final verdict {final_verdict.value} "
            f"with confidence {final_conf:.4f}."
        )
        explanation = VerificationExplanation(
            reasoning=reasoning_text,
            supporting_evidence_references=tuple(supporting),
            confidence_explanation=f"Selected max confidence matching threshold rules (Supp: {max_supp_conf:.2f}, Contra: {max_contra_conf:.2f}).",
            uncertainty_explanation=f"Estimated uncertainty is {1.0 - final_conf:.2f}.",
        )

        return VerificationResult(
            verdict=final_verdict,
            confidence=final_conf,
            supporting_passages=tuple(supporting),
            contradicting_passages=tuple(contradicting),
            evidence_attribution=attribution,
            explanation=explanation,
            uncertainty=float(1.0 - final_conf),
            model_metadata=model_metadata,
        )
