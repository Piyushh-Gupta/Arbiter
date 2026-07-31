"""Aggregation strategy abstractions and concrete implementations for verification results."""

from typing import Any, Protocol, Sequence, runtime_checkable

from src.core.verification.base import BaseEvidenceWeigher
from src.core.verification.verification_models import (
    AggregationTrace,
    ClaimVerificationInput,
    ConflictAnalysis,
    PassageVerificationResult,
    VerificationExplanation,
    VerificationModelMetadata,
    VerificationResult,
    VerificationVerdict,
    VerifierRuntimeMetadata,
)


@runtime_checkable
class BaseAggregationStrategy(Protocol):
    """Protocol for aggregating passage-level verification outcomes into claim-level verification results."""

    def aggregate(
        self,
        verification_input: ClaimVerificationInput,
        passage_results: tuple[PassageVerificationResult, ...],
        runtime_metadata: VerifierRuntimeMetadata | None = None,
    ) -> VerificationResult:
        """
        Aggregates passage verification results deterministically.

        Receives:
        - verification_input: Input specifications for claim verification.
        - passage_results: Tuple of PassageVerificationResult objects.
        - runtime_metadata: Optional model runtime metadata.

        Returns:
        - VerificationResult: Immutable final claim-level verdict.
        """
        ...


class DefaultEvidenceWeigher(BaseEvidenceWeigher):
    """Default evidence weigher combining retrieval score and verifier confidence."""

    def compute_weight(
        self,
        passage_result: PassageVerificationResult,
        passage_score: float,
    ) -> float:
        return float(passage_score) * float(passage_result.confidence)


def sort_passage_results(
    passage_results: Sequence[PassageVerificationResult],
    bundle_passages: Sequence[Any],
) -> list[PassageVerificationResult]:
    """Deterministically orders passage results by retrieval rank, verifier confidence (descending), and span ID (ascending)."""
    rank_map = {p.span_id: idx for idx, p in enumerate(bundle_passages)}

    def sort_key(pr: PassageVerificationResult) -> tuple[int, float, str]:
        rank = rank_map.get(pr.span_id, len(bundle_passages))
        conf = -float(pr.confidence)
        span_id = pr.span_id
        return rank, conf, span_id

    return sorted(passage_results, key=sort_key)


class MaxConfidenceAggregationStrategy(BaseAggregationStrategy):
    """
    Max-confidence aggregation strategy.
    Determines claim verdict based on maximum confidence across supporting vs contradicting passages,
    thresholded by definition rules.
    """

    def __init__(self, evidence_weigher: BaseEvidenceWeigher | None = None) -> None:
        self.evidence_weigher = evidence_weigher or DefaultEvidenceWeigher()

    def aggregate(
        self,
        verification_input: ClaimVerificationInput,
        passage_results: tuple[PassageVerificationResult, ...],
        runtime_metadata: VerifierRuntimeMetadata | None = None,
    ) -> VerificationResult:
        definition = verification_input.definition

        if runtime_metadata is None:
            model_metadata = VerificationModelMetadata(
                model_identifier=definition.verifier_model
            )
        else:
            model_metadata = VerificationModelMetadata(
                model_identifier=runtime_metadata.model_id,
                revision=runtime_metadata.revision,
                tokenizer=runtime_metadata.tokenizer,
                execution_device=str(runtime_metadata.execution_device),
                framework_version=runtime_metadata.framework,
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

        # 1. Deterministic sort before aggregation
        ordered_results = sort_passage_results(
            passage_results, verification_input.bundle.passages
        )

        passage_scores = {
            p.span_id: getattr(p, "score", 0.0)
            for p in verification_input.bundle.passages
        }

        supporting: list[str] = []
        contradicting: list[str] = []
        attribution: dict[str, float] = {}

        max_supp_conf = -1.0
        max_contra_conf = -1.0
        max_insuff_conf = -1.0

        weighting_decisions = {}

        for p_res in ordered_results:
            p_score = passage_scores.get(p_res.span_id, 0.0)
            w = self.evidence_weigher.compute_weight(p_res, p_score)
            weighting_decisions[p_res.span_id] = w
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

        insufficient = [
            pr.span_id
            for pr in ordered_results
            if pr.verdict == VerificationVerdict.INSUFFICIENT
        ]

        severity = 0.0
        if max_supp_conf > 0.0 and max_contra_conf > 0.0:
            severity = min(max_supp_conf, max_contra_conf) / max(
                max_supp_conf, max_contra_conf
            )
        imbalance = abs(max_supp_conf - max_contra_conf)

        conflict_analysis = ConflictAnalysis(
            supporting_passages=tuple(supporting),
            contradicting_passages=tuple(contradicting),
            insufficient_passages=tuple(insufficient),
            conflict_severity=severity,
            confidence_imbalance=imbalance,
            resolution_rationale=f"Resolved via MaxConfidence. Winner: {final_verdict}.",
        )

        aggregation_trace = AggregationTrace(
            aggregation_strategy="MAX_CONFIDENCE",
            ordered_evaluation_sequence=tuple(pr.span_id for pr in ordered_results),
            weighting_decisions=weighting_decisions,
            intermediate_scores={
                "max_supporting_confidence": max_supp_conf,
                "max_contradicting_confidence": max_contra_conf,
                "max_insufficient_confidence": max_insuff_conf,
            },
            final_decision_path=f"MaxConfidence determined {final_verdict} with confidence {final_conf}.",
        )

        reasoning_text = (
            f"Evaluated {len(ordered_results)} passage(s). Final verdict {final_verdict.value} "
            f"with confidence {final_conf:.4f}."
        )
        explanation = VerificationExplanation(
            reasoning=reasoning_text,
            supporting_evidence_references=tuple(supporting),
            confidence_explanation=(
                f"Selected max confidence matching threshold rules "
                f"(Supp: {max_supp_conf:.2f}, Contra: {max_contra_conf:.2f})."
            ),
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
            aggregation_trace=aggregation_trace,
            conflict_analysis=conflict_analysis,
        )
