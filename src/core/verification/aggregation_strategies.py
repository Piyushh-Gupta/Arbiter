"""Stateless multi-evidence aggregation strategies."""

from src.core.verification.aggregation import (
    BaseAggregationStrategy,
    DefaultEvidenceWeigher,
    sort_passage_results,
)
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


class WeightedVotingAggregationStrategy(BaseAggregationStrategy):
    """
    Weighted voting aggregation strategy.
    Sum of weights determined by BaseEvidenceWeigher determines the winner.
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

        ordered_results = sort_passage_results(
            passage_results, verification_input.bundle.passages
        )

        passage_scores = {
            p.span_id: getattr(p, "score", 0.0)
            for p in verification_input.bundle.passages
        }

        weighting_decisions = {}
        weighted_scores = {
            VerificationVerdict.SUPPORTED: 0.0,
            VerificationVerdict.CONTRADICTED: 0.0,
            VerificationVerdict.INSUFFICIENT: 0.0,
        }
        for pr in ordered_results:
            p_score = passage_scores.get(pr.span_id, 0.0)
            w = self.evidence_weigher.compute_weight(pr, p_score)
            weighting_decisions[pr.span_id] = w
            weighted_scores[pr.verdict] += w

        w_supp = weighted_scores[VerificationVerdict.SUPPORTED]
        w_contra = weighted_scores[VerificationVerdict.CONTRADICTED]
        w_insuff = weighted_scores[VerificationVerdict.INSUFFICIENT]

        if w_supp >= w_contra and w_supp >= w_insuff:
            candidate_verdict = VerificationVerdict.SUPPORTED
            winning_weight = w_supp
        elif w_contra > w_supp and w_contra >= w_insuff:
            candidate_verdict = VerificationVerdict.CONTRADICTED
            winning_weight = w_contra
        else:
            candidate_verdict = VerificationVerdict.INSUFFICIENT
            winning_weight = w_insuff

        total_weight = w_supp + w_contra + w_insuff
        final_conf = (winning_weight / total_weight) if total_weight > 0.0 else 0.0

        supp_thresh = definition.confidence_thresholds.get("SUPPORTED", 0.5)
        contra_thresh = definition.confidence_thresholds.get("CONTRADICTED", 0.5)

        if (
            candidate_verdict == VerificationVerdict.SUPPORTED
            and final_conf < supp_thresh
        ):
            final_verdict = VerificationVerdict.INSUFFICIENT
        elif (
            candidate_verdict == VerificationVerdict.CONTRADICTED
            and final_conf < contra_thresh
        ):
            final_verdict = VerificationVerdict.INSUFFICIENT
        else:
            final_verdict = candidate_verdict

        supporting = [
            pr.span_id
            for pr in ordered_results
            if pr.verdict == VerificationVerdict.SUPPORTED
        ]
        contradicting = [
            pr.span_id
            for pr in ordered_results
            if pr.verdict == VerificationVerdict.CONTRADICTED
        ]
        insufficient = [
            pr.span_id
            for pr in ordered_results
            if pr.verdict == VerificationVerdict.INSUFFICIENT
        ]

        severity = 0.0
        if w_supp > 0.0 and w_contra > 0.0:
            severity = min(w_supp, w_contra) / max(w_supp, w_contra)
        imbalance = abs(w_supp - w_contra)

        conflict_analysis = ConflictAnalysis(
            supporting_passages=tuple(supporting),
            contradicting_passages=tuple(contradicting),
            insufficient_passages=tuple(insufficient),
            conflict_severity=severity,
            confidence_imbalance=imbalance,
            resolution_rationale=(
                f"Weighted voting resolved to candidate {candidate_verdict.value} "
                f"(final {final_verdict.value}). Weight Sums - Supp: {w_supp:.4f}, "
                f"Contra: {w_contra:.4f}, Insuff: {w_insuff:.4f}."
            ),
        )

        aggregation_trace = AggregationTrace(
            aggregation_strategy="WEIGHTED_VOTING",
            ordered_evaluation_sequence=tuple(pr.span_id for pr in ordered_results),
            weighting_decisions=weighting_decisions,
            intermediate_scores={
                "weight_supported": w_supp,
                "weight_contradicted": w_contra,
                "weight_insufficient": w_insuff,
            },
            final_decision_path=(
                f"WeightedVoting selected candidate {candidate_verdict} (total weights: "
                f"{total_weight}). Final verdict {final_verdict}."
            ),
        )

        attribution = {
            pr.span_id: weighting_decisions[pr.span_id] for pr in ordered_results
        }

        explanation = VerificationExplanation(
            reasoning=(
                f"Weighted voting aggregation completed. Final verdict {final_verdict.value} "
                f"with confidence {final_conf:.4f}."
            ),
            supporting_evidence_references=tuple(supporting),
            confidence_explanation=f"Based on relative sum of weights. Total weight: {total_weight:.4f}.",
            uncertainty_explanation=f"Estimated uncertainty is {1.0 - final_conf:.2f}.",
        )

        return VerificationResult(
            verdict=final_verdict,
            confidence=final_conf,
            supporting_passages=tuple(supporting),
            contradicting_passages=tuple(contradicting),
            evidence_attribution=attribution,
            explanation=explanation,
            uncertainty=1.0 - final_conf,
            model_metadata=model_metadata,
            aggregation_trace=aggregation_trace,
            conflict_analysis=conflict_analysis,
        )


class ConsensusAggregationStrategy(BaseAggregationStrategy):
    """
    Consensus aggregation strategy.
    Requires agreement across passages above a threshold, else returns INSUFFICIENT.
    """

    def __init__(
        self,
        evidence_weigher: BaseEvidenceWeigher | None = None,
        consensus_threshold: float = 0.6,
    ) -> None:
        self.evidence_weigher = evidence_weigher or DefaultEvidenceWeigher()
        self.consensus_threshold = consensus_threshold

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

        ordered_results = sort_passage_results(
            passage_results, verification_input.bundle.passages
        )

        passage_scores = {
            p.span_id: getattr(p, "score", 0.0)
            for p in verification_input.bundle.passages
        }

        weighting_decisions = {}
        for pr in ordered_results:
            p_score = passage_scores.get(pr.span_id, 0.0)
            weighting_decisions[pr.span_id] = self.evidence_weigher.compute_weight(
                pr, p_score
            )

        counts = {
            VerificationVerdict.SUPPORTED: 0,
            VerificationVerdict.CONTRADICTED: 0,
            VerificationVerdict.INSUFFICIENT: 0,
        }
        for pr in ordered_results:
            counts[pr.verdict] += 1

        total_count = len(ordered_results)

        c_supp = counts[VerificationVerdict.SUPPORTED]
        c_contra = counts[VerificationVerdict.CONTRADICTED]
        c_insuff = counts[VerificationVerdict.INSUFFICIENT]

        if c_supp >= c_contra and c_supp >= c_insuff:
            majority_verdict = VerificationVerdict.SUPPORTED
            majority_count = c_supp
        elif c_contra > c_supp and c_contra >= c_insuff:
            majority_verdict = VerificationVerdict.CONTRADICTED
            majority_count = c_contra
        else:
            majority_verdict = VerificationVerdict.INSUFFICIENT
            majority_count = c_insuff

        majority_ratio = majority_count / total_count
        reached_consensus = majority_ratio >= self.consensus_threshold

        if reached_consensus:
            final_verdict = majority_verdict
            matching_passages = [
                pr for pr in ordered_results if pr.verdict == majority_verdict
            ]
            final_conf = sum(pr.confidence for pr in matching_passages) / len(
                matching_passages
            )
        else:
            final_verdict = VerificationVerdict.INSUFFICIENT
            final_conf = 0.0

        supporting = [
            pr.span_id
            for pr in ordered_results
            if pr.verdict == VerificationVerdict.SUPPORTED
        ]
        contradicting = [
            pr.span_id
            for pr in ordered_results
            if pr.verdict == VerificationVerdict.CONTRADICTED
        ]
        insufficient = [
            pr.span_id
            for pr in ordered_results
            if pr.verdict == VerificationVerdict.INSUFFICIENT
        ]

        severity = 0.0
        if c_supp > 0 and c_contra > 0:
            severity = min(c_supp, c_contra) / max(c_supp, c_contra)
        imbalance = abs(c_supp - c_contra) / total_count

        conflict_analysis = ConflictAnalysis(
            supporting_passages=tuple(supporting),
            contradicting_passages=tuple(contradicting),
            insufficient_passages=tuple(insufficient),
            conflict_severity=severity,
            confidence_imbalance=imbalance,
            resolution_rationale=(
                f"Consensus aggregation: reached={reached_consensus} "
                f"(ratio: {majority_ratio:.2f} >= threshold: {self.consensus_threshold:.2f}). "
                f"Majority: {majority_verdict.value}."
            ),
        )

        aggregation_trace = AggregationTrace(
            aggregation_strategy="CONSENSUS",
            ordered_evaluation_sequence=tuple(pr.span_id for pr in ordered_results),
            weighting_decisions=weighting_decisions,
            intermediate_scores={
                "count_supported": c_supp,
                "count_contradicted": c_contra,
                "count_insufficient": c_insuff,
                "majority_ratio": majority_ratio,
                "reached_consensus": reached_consensus,
            },
            final_decision_path=(
                f"Consensus majority {majority_verdict} (ratio {majority_ratio}). "
                f"Reached consensus: {reached_consensus}. Final verdict {final_verdict}."
            ),
        )

        explanation = VerificationExplanation(
            reasoning=(
                f"Consensus aggregation completed. reached={reached_consensus}. "
                f"Final verdict {final_verdict.value} with confidence {final_conf:.4f}."
            ),
            supporting_evidence_references=tuple(supporting),
            confidence_explanation=f"Average confidence of matching passages: {final_conf:.4f}.",
            uncertainty_explanation=f"Estimated uncertainty is {1.0 - final_conf:.2f}.",
        )

        attribution = {
            pr.span_id: weighting_decisions[pr.span_id] for pr in ordered_results
        }

        return VerificationResult(
            verdict=final_verdict,
            confidence=final_conf,
            supporting_passages=tuple(supporting),
            contradicting_passages=tuple(contradicting),
            evidence_attribution=attribution,
            explanation=explanation,
            uncertainty=1.0 - final_conf,
            model_metadata=model_metadata,
            aggregation_trace=aggregation_trace,
            conflict_analysis=conflict_analysis,
        )


class ContradictionAwareAggregationStrategy(BaseAggregationStrategy):
    """
    Contradiction-aware aggregation strategy.
    Consumes ConflictAnalysis to resolve conflicts deterministically.
    """

    def __init__(
        self,
        evidence_weigher: BaseEvidenceWeigher | None = None,
        contradiction_threshold: float = 0.3,
    ) -> None:
        self.evidence_weigher = evidence_weigher or DefaultEvidenceWeigher()
        self.contradiction_threshold = contradiction_threshold

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

        ordered_results = sort_passage_results(
            passage_results, verification_input.bundle.passages
        )

        passage_scores = {
            p.span_id: getattr(p, "score", 0.0)
            for p in verification_input.bundle.passages
        }

        weighting_decisions = {}
        for pr in ordered_results:
            p_score = passage_scores.get(pr.span_id, 0.0)
            weighting_decisions[pr.span_id] = self.evidence_weigher.compute_weight(
                pr, p_score
            )

        supporting = [
            pr.span_id
            for pr in ordered_results
            if pr.verdict == VerificationVerdict.SUPPORTED
        ]
        contradicting = [
            pr.span_id
            for pr in ordered_results
            if pr.verdict == VerificationVerdict.CONTRADICTED
        ]
        insufficient = [
            pr.span_id
            for pr in ordered_results
            if pr.verdict == VerificationVerdict.INSUFFICIENT
        ]

        max_supp_conf = -1.0
        max_contra_conf = -1.0
        max_insuff_conf = -1.0

        for pr in ordered_results:
            if pr.verdict == VerificationVerdict.SUPPORTED:
                if pr.confidence > max_supp_conf:
                    max_supp_conf = pr.confidence
            elif pr.verdict == VerificationVerdict.CONTRADICTED:
                if pr.confidence > max_contra_conf:
                    max_contra_conf = pr.confidence
            else:
                if pr.confidence > max_insuff_conf:
                    max_insuff_conf = pr.confidence

        severity = 0.0
        if max_supp_conf > 0.0 and max_contra_conf > 0.0:
            severity = min(max_supp_conf, max_contra_conf) / max(
                max_supp_conf, max_contra_conf
            )
        imbalance = abs(max_supp_conf - max_contra_conf)

        has_contradiction = (
            len(supporting) > 0
            and len(contradicting) > 0
            and max_supp_conf >= self.contradiction_threshold
            and max_contra_conf >= self.contradiction_threshold
        )

        if has_contradiction and imbalance < 0.15:
            final_verdict = VerificationVerdict.INSUFFICIENT
            final_conf = max(0.0, min(1.0, max(max_supp_conf, max_contra_conf) * 0.5))
            rationale = (
                f"Severe contradiction detected (imbalance: {imbalance:.2f} < 0.15). "
                f"Resolved to INSUFFICIENT."
            )
        else:
            if max_supp_conf >= max_contra_conf and max_supp_conf >= max_insuff_conf:
                final_verdict = VerificationVerdict.SUPPORTED
                final_conf = max_supp_conf
            elif max_contra_conf > max_supp_conf and max_contra_conf >= max_insuff_conf:
                final_verdict = VerificationVerdict.CONTRADICTED
                final_conf = max_contra_conf
            else:
                final_verdict = VerificationVerdict.INSUFFICIENT
                final_conf = max_insuff_conf
            rationale = f"No severe contradiction or one category dominated. Winner: {final_verdict}."

        conflict_analysis = ConflictAnalysis(
            supporting_passages=tuple(supporting),
            contradicting_passages=tuple(contradicting),
            insufficient_passages=tuple(insufficient),
            conflict_severity=severity,
            confidence_imbalance=imbalance,
            resolution_rationale=rationale,
        )

        aggregation_trace = AggregationTrace(
            aggregation_strategy="CONTRADICTION_AWARE",
            ordered_evaluation_sequence=tuple(pr.span_id for pr in ordered_results),
            weighting_decisions=weighting_decisions,
            intermediate_scores={
                "max_supporting_confidence": max_supp_conf,
                "max_contradicting_confidence": max_contra_conf,
                "max_insufficient_confidence": max_insuff_conf,
                "has_contradiction": has_contradiction,
                "imbalance": imbalance,
            },
            final_decision_path=f"ContradictionAware path: {rationale} Final verdict {final_verdict}.",
        )

        explanation = VerificationExplanation(
            reasoning=rationale,
            supporting_evidence_references=tuple(supporting),
            confidence_explanation=f"Confidence determined by active branch: {final_conf:.4f}.",
            uncertainty_explanation=f"Estimated uncertainty is {1.0 - final_conf:.2f}.",
        )

        attribution = {
            pr.span_id: weighting_decisions[pr.span_id] for pr in ordered_results
        }

        return VerificationResult(
            verdict=final_verdict,
            confidence=final_conf,
            supporting_passages=tuple(supporting),
            contradicting_passages=tuple(contradicting),
            evidence_attribution=attribution,
            explanation=explanation,
            uncertainty=1.0 - final_conf,
            model_metadata=model_metadata,
            aggregation_trace=aggregation_trace,
            conflict_analysis=conflict_analysis,
        )
