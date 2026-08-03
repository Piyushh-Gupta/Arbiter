"""Stateless top-level pipeline orchestrator."""

import hashlib
import time
from datetime import datetime, timezone
from typing import Any

from src.core.evaluation.evaluation_models import EvaluationResult
from src.core.exceptions import PipelineConfigurationError, PipelineStageExecutionError
from src.core.pipeline.base import BasePipelineOrchestrator, BasePipelineStage
from src.core.pipeline.pipeline_models import (
    PipelineDefinition,
    PipelineExecutionContext,
    PipelineExecutionRequest,
    PipelineExecutionResult,
    PipelineRuntimeMetadata,
    PipelineStageDefinition,
    PipelineStageMetadata,
)
from src.core.pipeline.profile_models import PipelineProfileRegistry, PipelineStageRegistry
from src.core.retrieval.retriever import ClaimRetriever
from src.core.verification.verifier import ClaimVerifier
from src.core.failure_analysis.analyzer import FailureAnalyzer
from src.core.uncertainty.estimator import UncertaintyEstimator
from src.core.decision.engine import DecisionEngine
from src.core.explainability.explainer import Explainer
from src.core.evaluation.evaluator import Evaluator


class RetrievalStage(BasePipelineStage):
    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self._retriever = ClaimRetriever()

    def validate_compatibility(self, definition: PipelineStageDefinition) -> None:
        try:
            self._registry.resolve(definition.profile_id)
        except Exception as e:
            raise PipelineConfigurationError(f"Retrieval stage error: {e}") from e

    def execute(self, claim: str, profile_id: str) -> Any:
        prof = self._registry.resolve(profile_id)
        return self._retriever.retrieve(claim, prof.definition, prof.strategy)


class VerificationStage(BasePipelineStage):
    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self._verifier = ClaimVerifier()

    def validate_compatibility(self, definition: PipelineStageDefinition) -> None:
        try:
            self._registry.resolve(definition.profile_id)
        except Exception as e:
            raise PipelineConfigurationError(f"Verification stage error: {e}") from e

    def execute(self, claim: str, ret_res: Any, profile_id: str) -> Any:
        prof = self._registry.resolve(profile_id)
        return self._verifier.verify(claim, ret_res, prof.definition, prof.verifier)


class FailureAnalysisStage(BasePipelineStage):
    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self._analyzer = FailureAnalyzer()

    def validate_compatibility(self, definition: PipelineStageDefinition) -> None:
        try:
            self._registry.resolve(definition.profile_id)
        except Exception as e:
            raise PipelineConfigurationError(f"Failure Analysis stage error: {e}") from e

    def execute(self, claim: str, ver_res: Any, profile_id: str) -> Any:
        prof = self._registry.resolve(profile_id)
        return self._analyzer.analyze(claim, ver_res, prof.definition, prof.analyzer)


class UncertaintyStage(BasePipelineStage):
    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self._estimator = UncertaintyEstimator()

    def validate_compatibility(self, definition: PipelineStageDefinition) -> None:
        try:
            self._registry.resolve(definition.profile_id)
        except Exception as e:
            raise PipelineConfigurationError(f"Uncertainty stage error: {e}") from e

    def execute(self, claim: str, fa_res: Any, profile_id: str) -> Any:
        prof = self._registry.resolve(profile_id)
        return self._estimator.estimate(claim, fa_res, prof.definition, prof.estimator)


class DecisionStage(BasePipelineStage):
    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self._engine = DecisionEngine()

    def validate_compatibility(self, definition: PipelineStageDefinition) -> None:
        try:
            self._registry.resolve(definition.profile_id)
        except Exception as e:
            raise PipelineConfigurationError(f"Decision stage error: {e}") from e

    def execute(self, claim: str, unc_res: Any, profile_id: str) -> Any:
        prof = self._registry.resolve(profile_id)
        return self._engine.decide(claim, unc_res, prof.definition, prof.engine)


class ExplanationStage(BasePipelineStage):
    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self._explainer = Explainer()

    def validate_compatibility(self, definition: PipelineStageDefinition) -> None:
        try:
            self._registry.resolve(definition.profile_id)
        except Exception as e:
            raise PipelineConfigurationError(f"Explanation stage error: {e}") from e

    def execute(self, claim: str, dec_res: Any, profile_id: str) -> Any:
        prof = self._registry.resolve(profile_id)
        return self._explainer.explain(claim, dec_res, prof.definition, prof.engine)


class EvaluationStage(BasePipelineStage):
    def __init__(self, registry: Any) -> None:
        self._registry = registry
        self._evaluator = Evaluator()

    def validate_compatibility(self, definition: PipelineStageDefinition) -> None:
        try:
            self._registry.resolve(definition.profile_id)
        except Exception as e:
            raise PipelineConfigurationError(f"Evaluation stage error: {e}") from e

    def execute(self, exp_res: Any, profile_id: str) -> Any:
        prof = self._registry.resolve(profile_id)
        return self._evaluator.evaluate(exp_res, prof.definition, prof.engine)


class ModernArbiterPipeline(BasePipelineOrchestrator):
    """Stateless orchestrator executing stage abstractions via profiles."""

    def __init__(
        self,
        stage_registry: PipelineStageRegistry,
    ) -> None:
        self._pipeline_registry: PipelineProfileRegistry | None = None
        self._stage_registry = stage_registry
        self._pipeline_version = "1.0.0"
        self._schema_version = "1.0.0"
        self._execution_environment = "production"

    def set_pipeline_registry(self, registry: PipelineProfileRegistry) -> None:
        self._pipeline_registry = registry

    def validate_compatibility(self, definition: PipelineDefinition) -> None:
        try:
            self._stage_registry.resolve(definition.retrieval_stage.profile_id)
            self._stage_registry.resolve(definition.verification_stage.profile_id)
            self._stage_registry.resolve(definition.failure_analysis_stage.profile_id)
            self._stage_registry.resolve(definition.uncertainty_stage.profile_id)
            self._stage_registry.resolve(definition.decision_stage.profile_id)
            self._stage_registry.resolve(definition.explanation_stage.profile_id)
            self._stage_registry.resolve(definition.evaluation_stage.profile_id)
        except Exception as e:
            raise PipelineConfigurationError(f"Invalid pipeline stage binding: {e}") from e

    def execute(self, request: PipelineExecutionRequest) -> PipelineExecutionResult:
        if self._pipeline_registry is None:
            raise PipelineConfigurationError("Pipeline registry not set on orchestrator.")
        pipeline_profile = self._pipeline_registry.resolve(request.pipeline_profile_id)
        definition = pipeline_profile.definition

        fingerprint = hashlib.sha256(definition.pipeline_id.encode("utf-8")).hexdigest()
        execution_id = hashlib.sha256(f"{definition.pipeline_id}{request.claim}".encode("utf-8")).hexdigest()
        
        runtime_metadata = PipelineRuntimeMetadata(
            pipeline_version=self._pipeline_version,
            configuration_fingerprint=fingerprint,
            schema_version=self._schema_version,
            execution_environment=self._execution_environment,
            execution_timestamp=datetime.now(timezone.utc),
        )

        stage_metadata_list = []
        total_start = time.perf_counter()
        
        try:
            # 1. Retrieval
            ret_def = definition.retrieval_stage
            ret_prof = self._stage_registry.resolve(ret_def.profile_id)
            t0 = time.perf_counter()
            ret_res = ret_prof.stage.execute(request.claim, profile_id=ret_def.profile_id)
            stage_metadata_list.append(PipelineStageMetadata(
                stage_id=ret_def.stage_id, profile_id=ret_def.profile_id,
                latency_ms=(time.perf_counter() - t0) * 1000, success=True
            ))

            # 2. Verification
            ver_def = definition.verification_stage
            ver_prof = self._stage_registry.resolve(ver_def.profile_id)
            t0 = time.perf_counter()
            ver_res = ver_prof.stage.execute(request.claim, ret_res, profile_id=ver_def.profile_id)
            stage_metadata_list.append(PipelineStageMetadata(
                stage_id=ver_def.stage_id, profile_id=ver_def.profile_id,
                latency_ms=(time.perf_counter() - t0) * 1000, success=True
            ))

            # 3. Failure Analysis
            fa_def = definition.failure_analysis_stage
            fa_prof = self._stage_registry.resolve(fa_def.profile_id)
            t0 = time.perf_counter()
            fa_res = fa_prof.stage.execute(request.claim, ver_res, profile_id=fa_def.profile_id)
            stage_metadata_list.append(PipelineStageMetadata(
                stage_id=fa_def.stage_id, profile_id=fa_def.profile_id,
                latency_ms=(time.perf_counter() - t0) * 1000, success=True
            ))

            # 4. Uncertainty
            unc_def = definition.uncertainty_stage
            unc_prof = self._stage_registry.resolve(unc_def.profile_id)
            t0 = time.perf_counter()
            unc_res = unc_prof.stage.execute(request.claim, fa_res, profile_id=unc_def.profile_id)
            stage_metadata_list.append(PipelineStageMetadata(
                stage_id=unc_def.stage_id, profile_id=unc_def.profile_id,
                latency_ms=(time.perf_counter() - t0) * 1000, success=True
            ))

            # 5. Decision
            dec_def = definition.decision_stage
            dec_prof = self._stage_registry.resolve(dec_def.profile_id)
            t0 = time.perf_counter()
            dec_res = dec_prof.stage.execute(request.claim, unc_res, profile_id=dec_def.profile_id)
            stage_metadata_list.append(PipelineStageMetadata(
                stage_id=dec_def.stage_id, profile_id=dec_def.profile_id,
                latency_ms=(time.perf_counter() - t0) * 1000, success=True
            ))

            # 6. Explanation
            exp_def = definition.explanation_stage
            exp_prof = self._stage_registry.resolve(exp_def.profile_id)
            t0 = time.perf_counter()
            exp_res = exp_prof.stage.execute(request.claim, dec_res, profile_id=exp_def.profile_id)
            stage_metadata_list.append(PipelineStageMetadata(
                stage_id=exp_def.stage_id, profile_id=exp_def.profile_id,
                latency_ms=(time.perf_counter() - t0) * 1000, success=True
            ))

            # 7. Evaluation
            eval_def = definition.evaluation_stage
            eval_prof = self._stage_registry.resolve(eval_def.profile_id)
            t0 = time.perf_counter()
            eval_res = eval_prof.stage.execute(exp_res, profile_id=eval_def.profile_id)
            stage_metadata_list.append(PipelineStageMetadata(
                stage_id=eval_def.stage_id, profile_id=eval_def.profile_id,
                latency_ms=(time.perf_counter() - t0) * 1000, success=True
            ))

            context = PipelineExecutionContext(
                execution_id=execution_id,
                pipeline_id=definition.pipeline_id,
                claim=request.claim,
                runtime_metadata=runtime_metadata,
                stage_metadata=tuple(stage_metadata_list),
                total_latency_ms=(time.perf_counter() - total_start) * 1000,
                success=True,
            )
            return PipelineExecutionResult(
                evaluation_result=eval_res,
                execution_context=context,
            )
        except Exception as e:
            # record failure
            context = PipelineExecutionContext(
                execution_id=execution_id,
                pipeline_id=definition.pipeline_id,
                claim=request.claim,
                runtime_metadata=runtime_metadata,
                stage_metadata=tuple(stage_metadata_list),
                total_latency_ms=(time.perf_counter() - total_start) * 1000,
                success=False,
            )
            raise PipelineStageExecutionError(f"Pipeline execution failed: {e}") from e


class ArbiterPipeline:
    """Legacy backward-compatible adapter."""
    
    def __init__(
        self,
        retrieval_registry: Any = None,
        verification_registry: Any = None,
        failure_analysis_registry: Any = None,
        uncertainty_registry: Any = None,
        decision_registry: Any = None,
        explanation_registry: Any = None,
        evaluation_registry: Any = None,
        modern_pipeline: ModernArbiterPipeline | None = None,
    ) -> None:
        self.modern_pipeline = modern_pipeline

    def execute(self, request: PipelineExecutionRequest) -> EvaluationResult:
        if self.modern_pipeline is None:
            raise PipelineConfigurationError("Modern pipeline adapter missing modern_pipeline instance.")
        return self.modern_pipeline.execute(request).evaluation_result
