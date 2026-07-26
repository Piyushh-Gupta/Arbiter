"""Unit tests for the M15.1 Pipeline Orchestrator."""

import pytest
from pydantic import ValidationError

from src.core.decision.base import BaseDecisionEngine
from src.core.decision.decision_models import (
    DecisionAction,
    DecisionDefinition,
    DecisionMetadata,
    DecisionProfile,
    DecisionProfileRegistry,
    DecisionResult,
)
from src.core.evaluation.base import BaseEvaluator
from src.core.evaluation.evaluation_models import (
    EvaluationDefinition,
    EvaluationMetadata,
    EvaluationMetric,
    EvaluationProfile,
    EvaluationProfileRegistry,
    EvaluationResult,
)
from src.core.exceptions import (
    DecisionProfileNotFoundError,
    VerificationProfileNotFoundError,
)
from src.core.explainability.base import BaseExplainer
from src.core.explainability.explainability_models import (
    ExplanationDefinition,
    ExplanationMetadata,
    ExplanationProfile,
    ExplanationProfileRegistry,
    ExplanationResult,
    ExplanationSection,
)
from src.core.failure_analysis.base import BaseFailureAnalyzer
from src.core.failure_analysis.failure_analysis_models import (
    FailureAnalysisDefinition,
    FailureAnalysisProfile,
    FailureAnalysisProfileRegistry,
    FailureAnalysisResult,
    FailureMetadata,
    FailureSeverity,
)
from src.core.pipeline.orchestrator import ArbiterPipeline
from src.core.pipeline.pipeline_models import PipelineExecutionRequest
from src.core.retrieval.base import BaseRetriever
from src.core.retrieval.retrieval_models import (
    EvidenceBundle,
    RetrievalDefinition,
    RetrievalMetadata,
    RetrievalProfile,
    RetrievalProfileRegistry,
)
from src.core.uncertainty.base import BaseUncertaintyEstimator
from src.core.uncertainty.uncertainty_models import (
    UncertaintyDefinition,
    UncertaintyLevel,
    UncertaintyMetadata,
    UncertaintyProfile,
    UncertaintyProfileRegistry,
    UncertaintyResult,
)
from src.core.verification.base import BaseVerifier
from src.core.verification.verification_models import (
    VerificationDefinition,
    VerificationLabel,
    VerificationMetadata,
    VerificationProfile,
    VerificationProfileRegistry,
    VerificationResult,
)


# Mock Implementations for isolated testing
class MockRetrievalDefinition(RetrievalDefinition):
    pass


class MockRetriever(BaseRetriever):
    def validate_compatibility(self, definition: RetrievalDefinition) -> None:
        pass

    def retrieve(self, claim: str, definition: RetrievalDefinition) -> EvidenceBundle:
        return EvidenceBundle(
            claim=claim,
            passages=(),
            metadata=RetrievalMetadata(strategy_id="mock", top_k=5),
        )


class MockVerificationDefinition(VerificationDefinition):
    pass


class MockVerifier(BaseVerifier):
    def validate_compatibility(self, definition: VerificationDefinition) -> None:
        pass

    def verify(
        self, claim: str, bundle: EvidenceBundle, definition: VerificationDefinition
    ) -> VerificationResult:
        return VerificationResult(
            label=VerificationLabel.SUPPORTS,
            evidence_bundle=bundle,
            metadata=VerificationMetadata(strategy_id="mock"),
        )


class MockFailureAnalysisDefinition(FailureAnalysisDefinition):
    pass


class MockFailureAnalyzer(BaseFailureAnalyzer):
    def validate_compatibility(self, definition: FailureAnalysisDefinition) -> None:
        pass

    def analyze(
        self,
        claim: str,
        verification_result: VerificationResult,
        definition: FailureAnalysisDefinition,
    ) -> FailureAnalysisResult:
        return FailureAnalysisResult(
            failure_flags=frozenset(),
            severity=FailureSeverity.NONE,
            verification_result=verification_result,
            metadata=FailureMetadata(strategy_id="mock"),
        )


class MockUncertaintyDefinition(UncertaintyDefinition):
    pass


class MockUncertaintyEstimator(BaseUncertaintyEstimator):
    def validate_compatibility(self, definition: UncertaintyDefinition) -> None:
        pass

    def estimate(
        self,
        claim: str,
        failure_analysis_result: FailureAnalysisResult,
        definition: UncertaintyDefinition,
    ) -> UncertaintyResult:
        return UncertaintyResult(
            level=UncertaintyLevel.LOW,
            score=0.1,
            factors=frozenset(),
            failure_analysis_result=failure_analysis_result,
            metadata=UncertaintyMetadata(strategy_id="mock"),
        )


class MockDecisionDefinition(DecisionDefinition):
    pass


class MockDecisionEngine(BaseDecisionEngine):
    def validate_compatibility(self, definition: DecisionDefinition) -> None:
        pass

    def decide(
        self,
        claim: str,
        uncertainty_result: UncertaintyResult,
        definition: DecisionDefinition,
    ) -> DecisionResult:
        return DecisionResult(
            action=DecisionAction.ACCEPT,
            rationale="mock",
            uncertainty_result=uncertainty_result,
            metadata=DecisionMetadata(strategy_id="mock"),
        )


class MockExplanationDefinition(ExplanationDefinition):
    pass


class MockExplainer(BaseExplainer):
    def validate_compatibility(self, definition: ExplanationDefinition) -> None:
        pass

    def explain(
        self,
        claim: str,
        decision_result: DecisionResult,
        definition: ExplanationDefinition,
    ) -> ExplanationResult:
        return ExplanationResult(
            sections=(
                ExplanationSection(identifier="mock", title="Mock", content="Mock"),
            ),
            decision_result=decision_result,
            metadata=ExplanationMetadata(strategy_id="mock"),
        )


class MockEvaluationDefinition(EvaluationDefinition):
    pass


class MockEvaluator(BaseEvaluator):
    def validate_compatibility(self, definition: EvaluationDefinition) -> None:
        pass

    def evaluate(
        self, explanation_result: ExplanationResult, definition: EvaluationDefinition
    ) -> EvaluationResult:
        return EvaluationResult(
            metrics=(EvaluationMetric(identifier="mock", title="Mock", score=1.0),),
            explanation_result=explanation_result,
            metadata=EvaluationMetadata(strategy_id="mock"),
        )


def build_valid_registries() -> (
    tuple[
        RetrievalProfileRegistry,
        VerificationProfileRegistry,
        FailureAnalysisProfileRegistry,
        UncertaintyProfileRegistry,
        DecisionProfileRegistry,
        ExplanationProfileRegistry,
        EvaluationProfileRegistry,
    ]
):
    ret_reg = RetrievalProfileRegistry(
        profiles=(
            RetrievalProfile(
                profile_id="p_ret",
                definition=MockRetrievalDefinition(),
                strategy=MockRetriever(),
            ),
        )
    )
    ver_reg = VerificationProfileRegistry(
        profiles=(
            VerificationProfile(
                profile_id="p_ver",
                definition=MockVerificationDefinition(),
                verifier=MockVerifier(),
            ),
        )
    )
    fa_reg = FailureAnalysisProfileRegistry(
        profiles=(
            FailureAnalysisProfile(
                profile_id="p_fa",
                definition=MockFailureAnalysisDefinition(),
                analyzer=MockFailureAnalyzer(),
            ),
        )
    )
    unc_reg = UncertaintyProfileRegistry(
        profiles=(
            UncertaintyProfile(
                profile_id="p_unc",
                definition=MockUncertaintyDefinition(),
                estimator=MockUncertaintyEstimator(),
            ),
        )
    )
    dec_reg = DecisionProfileRegistry(
        profiles=(
            DecisionProfile(
                profile_id="p_dec",
                definition=MockDecisionDefinition(),
                engine=MockDecisionEngine(),
            ),
        )
    )
    exp_reg = ExplanationProfileRegistry(
        profiles=(
            ExplanationProfile(
                profile_id="p_exp",
                definition=MockExplanationDefinition(),
                engine=MockExplainer(),
            ),
        )
    )
    eval_reg = EvaluationProfileRegistry(
        profiles=(
            EvaluationProfile(
                profile_id="p_eval",
                definition=MockEvaluationDefinition(),
                engine=MockEvaluator(),
            ),
        )
    )
    return ret_reg, ver_reg, fa_reg, unc_reg, dec_reg, exp_reg, eval_reg


def test_immutable_request() -> None:
    req = PipelineExecutionRequest(
        claim="test",
        retrieval_profile_id="1",
        verification_profile_id="2",
        failure_analysis_profile_id="3",
        uncertainty_profile_id="4",
        decision_profile_id="5",
        explanation_profile_id="6",
        evaluation_profile_id="7",
    )
    with pytest.raises(ValidationError):
        req.claim = "new"


def test_pipeline_execution_equivalence_and_identity() -> None:
    regs = build_valid_registries()
    pipeline = ArbiterPipeline(*regs)

    req = PipelineExecutionRequest(
        claim="Is it true?",
        retrieval_profile_id="p_ret",
        verification_profile_id="p_ver",
        failure_analysis_profile_id="p_fa",
        uncertainty_profile_id="p_unc",
        decision_profile_id="p_dec",
        explanation_profile_id="p_exp",
        evaluation_profile_id="p_eval",
    )

    eval_res = pipeline.execute(req)

    # Check that execution returned the correct type and identity chain holds
    assert eval_res.explanation_result is not None
    assert eval_res.explanation_result.decision_result is not None
    assert eval_res.explanation_result.decision_result.uncertainty_result is not None
    assert (
        eval_res.explanation_result.decision_result.uncertainty_result.failure_analysis_result
        is not None
    )
    assert (
        eval_res.explanation_result.decision_result.uncertainty_result.failure_analysis_result.verification_result
        is not None
    )
    assert (
        eval_res.explanation_result.decision_result.uncertainty_result.failure_analysis_result.verification_result.evidence_bundle
        is not None
    )

    # Initial claim is preserved at the very bottom
    assert (
        eval_res.explanation_result.decision_result.uncertainty_result.failure_analysis_result.verification_result.evidence_bundle.claim
        == "Is it true?"
    )

    # Evaluator ran successfully
    assert len(eval_res.metrics) == 1


def test_jit_unknown_profile_failure() -> None:
    regs = build_valid_registries()
    pipeline = ArbiterPipeline(*regs)

    # Invalid verification profile ID should throw VerificationProfileNotFoundError
    req1 = PipelineExecutionRequest(
        claim="test",
        retrieval_profile_id="p_ret",
        verification_profile_id="invalid_ver",
        failure_analysis_profile_id="p_fa",
        uncertainty_profile_id="p_unc",
        decision_profile_id="p_dec",
        explanation_profile_id="p_exp",
        evaluation_profile_id="p_eval",
    )

    with pytest.raises(
        VerificationProfileNotFoundError,
        match="Verification profile not found: invalid_ver",
    ):
        pipeline.execute(req1)

    # Invalid decision profile ID should throw DecisionProfileNotFoundError (but only after Verification works)
    req2 = PipelineExecutionRequest(
        claim="test",
        retrieval_profile_id="p_ret",
        verification_profile_id="p_ver",
        failure_analysis_profile_id="p_fa",
        uncertainty_profile_id="p_unc",
        decision_profile_id="invalid_dec",
        explanation_profile_id="p_exp",
        evaluation_profile_id="p_eval",
    )

    with pytest.raises(
        DecisionProfileNotFoundError, match="Decision profile not found: invalid_dec"
    ):
        pipeline.execute(req2)
