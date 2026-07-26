"""Evaluation endpoint routing."""

from fastapi import APIRouter, Request

from src.api.models.evaluation import (
    EvaluateClaimRequest,
    EvaluationResponse,
    MetricDTO,
)
from src.core.pipeline.orchestrator import ArbiterPipeline
from src.core.pipeline.pipeline_models import PipelineExecutionRequest

router = APIRouter()


@router.post(
    "/v1/evaluate",
    response_model=EvaluationResponse,
    summary="Evaluate Claim",
    description="Executes the full Arbiter pipeline to evaluate a claim.",
)
async def evaluate_claim(
    payload: EvaluateClaimRequest,
    request: Request,
) -> EvaluationResponse:
    """Executes the Arbiter pipeline."""
    # Retrieve the state-attached pipeline instance
    pipeline: ArbiterPipeline = request.app.state.pipeline

    # Map DTO -> Domain Model
    domain_req = PipelineExecutionRequest(
        claim=payload.claim,
        retrieval_profile_id=payload.retrieval_profile_id,
        verification_profile_id=payload.verification_profile_id,
        failure_analysis_profile_id=payload.failure_analysis_profile_id,
        uncertainty_profile_id=payload.uncertainty_profile_id,
        decision_profile_id=payload.decision_profile_id,
        explanation_profile_id=payload.explanation_profile_id,
        evaluation_profile_id=payload.evaluation_profile_id,
    )

    # Execute business logic statelessly
    domain_res = pipeline.execute(domain_req)

    # Map Domain Model -> DTO
    metrics = [
        MetricDTO(identifier=m.identifier, title=m.title, score=m.score)
        for m in domain_res.metrics
    ]

    return EvaluationResponse(metrics=metrics)
