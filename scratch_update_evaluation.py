with open("src/api/routes/evaluation.py", "w", encoding="utf-8") as f:
    f.write('''"""Evaluation endpoint routing."""

import uuid

from fastapi import APIRouter, Request

from src.api.models.evaluation import (
    EvaluateClaimRequest,
)
from src.api.services.registry import ServiceRegistry
from src.api.services.service_models import (
    ClientMetadata,
    EvaluationRequest,
    EvaluationResponse,
    RequestMetadata,
    ServiceContext,
)

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
    """Executes the Arbiter pipeline through the Service Layer."""
    registry: ServiceRegistry = request.app.state.service_registry
    
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    
    context = ServiceContext(
        correlation_id=correlation_id,
        request_metadata=RequestMetadata(
            headers=dict(request.headers),
            query_params=dict(request.query_params),
            client_ip=request.client.host if request.client else None,
        ),
        client_metadata=ClientMetadata(
            user_agent=request.headers.get("user-agent"),
        ),
    )
    
    req = EvaluationRequest(
        claim=payload.claim,
        pipeline_profile_id=payload.pipeline_profile_id,
        context=context,
    )
    
    return registry.evaluation_service.evaluate(req)
''')
