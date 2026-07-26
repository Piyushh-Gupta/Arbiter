"""Stateless top-level pipeline orchestrator."""

from src.core.decision.decision_models import DecisionProfileRegistry
from src.core.decision.engine import DecisionEngine
from src.core.evaluation.evaluation_models import (
    EvaluationProfileRegistry,
    EvaluationResult,
)
from src.core.evaluation.evaluator import Evaluator
from src.core.explainability.explainability_models import ExplanationProfileRegistry
from src.core.explainability.explainer import Explainer
from src.core.failure_analysis.analyzer import FailureAnalyzer
from src.core.failure_analysis.failure_analysis_models import (
    FailureAnalysisProfileRegistry,
)
from src.core.pipeline.pipeline_models import PipelineExecutionRequest
from src.core.retrieval.retrieval_models import RetrievalProfileRegistry
from src.core.retrieval.retriever import ClaimRetriever
from src.core.uncertainty.estimator import UncertaintyEstimator
from src.core.uncertainty.uncertainty_models import UncertaintyProfileRegistry
from src.core.verification.verification_models import VerificationProfileRegistry
from src.core.verification.verifier import ClaimVerifier


class ArbiterPipeline:
    """Stateless top-level orchestrator of the Arbiter pipeline."""

    def __init__(
        self,
        retrieval_registry: RetrievalProfileRegistry,
        verification_registry: VerificationProfileRegistry,
        failure_analysis_registry: FailureAnalysisProfileRegistry,
        uncertainty_registry: UncertaintyProfileRegistry,
        decision_registry: DecisionProfileRegistry,
        explanation_registry: ExplanationProfileRegistry,
        evaluation_registry: EvaluationProfileRegistry,
    ) -> None:
        self._retrieval_registry = retrieval_registry
        self._verification_registry = verification_registry
        self._failure_analysis_registry = failure_analysis_registry
        self._uncertainty_registry = uncertainty_registry
        self._decision_registry = decision_registry
        self._explanation_registry = explanation_registry
        self._evaluation_registry = evaluation_registry

        self._retriever = ClaimRetriever()
        self._verifier = ClaimVerifier()
        self._failure_analyzer = FailureAnalyzer()
        self._uncertainty_estimator = UncertaintyEstimator()
        self._decision_engine = DecisionEngine()
        self._explainer = Explainer()
        self._evaluator = Evaluator()

    def execute(self, request: PipelineExecutionRequest) -> EvaluationResult:
        """
        Executes the full pipeline deterministically, resolving profiles just-in-time.
        """
        # 1. Retrieval
        ret_prof = self._retrieval_registry.resolve(request.retrieval_profile_id)
        ret_res = self._retriever.retrieve(
            request.claim, ret_prof.definition, ret_prof.strategy
        )

        # 2. Verification
        ver_prof = self._verification_registry.resolve(request.verification_profile_id)
        ver_res = self._verifier.verify(
            request.claim, ret_res, ver_prof.definition, ver_prof.verifier
        )

        # 3. Failure Analysis
        fa_prof = self._failure_analysis_registry.resolve(
            request.failure_analysis_profile_id
        )
        fa_res = self._failure_analyzer.analyze(
            request.claim, ver_res, fa_prof.definition, fa_prof.analyzer
        )

        # 4. Uncertainty Estimation
        unc_prof = self._uncertainty_registry.resolve(request.uncertainty_profile_id)
        unc_res = self._uncertainty_estimator.estimate(
            request.claim, fa_res, unc_prof.definition, unc_prof.estimator
        )

        # 5. Decision
        dec_prof = self._decision_registry.resolve(request.decision_profile_id)
        dec_res = self._decision_engine.decide(
            request.claim, unc_res, dec_prof.definition, dec_prof.engine
        )

        # 6. Explainability
        exp_prof = self._explanation_registry.resolve(request.explanation_profile_id)
        exp_res = self._explainer.explain(
            request.claim, dec_res, exp_prof.definition, exp_prof.engine
        )

        # 7. Evaluation
        eval_prof = self._evaluation_registry.resolve(request.evaluation_profile_id)
        eval_res = self._evaluator.evaluate(
            exp_res, eval_prof.definition, eval_prof.engine
        )

        return eval_res
