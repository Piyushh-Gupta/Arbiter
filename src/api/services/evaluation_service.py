"""Evaluation service implementation."""

import time

from src.api.services.base import BaseEvaluationService
from src.api.services.service_models import (
    EvaluationRequest,
    EvaluationResponse,
    MetricServiceDTO,
    ServiceExecutionMetadata,
)
from src.core.pipeline.orchestrator import ArbiterPipeline
from src.core.pipeline.pipeline_models import PipelineExecutionRequest


class EvaluationService(BaseEvaluationService):
    """Stateless evaluation service."""

    def __init__(self, pipeline: ArbiterPipeline) -> None:
        self._pipeline = pipeline

    def evaluate(self, request: EvaluationRequest) -> EvaluationResponse:
        """Executes the evaluation request deterministically."""
        start_ns = time.perf_counter_ns()

        domain_req = PipelineExecutionRequest(
            claim=request.claim,
            pipeline_profile_id=request.pipeline_profile_id,
        )

        domain_res = self._pipeline.execute(domain_req)

        metrics = tuple(
            MetricServiceDTO(identifier=m.identifier, title=m.title, score=m.score)
            for m in domain_res.metrics
        )

        end_ns = time.perf_counter_ns()
        duration_ms = (end_ns - start_ns) / 1_000_000.0

        return EvaluationResponse(
            metrics=metrics,
            execution_metadata=ServiceExecutionMetadata(
                start_time_ns=start_ns,
                end_time_ns=end_ns,
                duration_ms=duration_ms,
            ),
            correlation_id=request.context.correlation_id,
        )
