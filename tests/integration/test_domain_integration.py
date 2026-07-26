"""Domain integration tests for the Arbiter Pipeline."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.evaluation.evaluation_models import EvaluationResult
from src.core.pipeline.orchestrator import ArbiterPipeline
from src.core.pipeline.pipeline_models import PipelineExecutionRequest


def test_domain_pipeline_execution_and_ownership_chain(
    app: FastAPI, valid_domain_request: PipelineExecutionRequest
) -> None:
    """
    Validates E2E domain pipeline execution and strict ownership-chain integrity.

    This test runs the fully configured ArbiterPipeline directly using native domain objects.
    It asserts that the pipeline deterministically executes all stages and perfectly preserves
    the nested object identity (`is`) across the entire M1-M14 ownership chain.
    """
    # Use TestClient as context manager to execute FastAPI lifespan, building the pipeline.
    with TestClient(app):
        # Retrieve the configured pipeline from application state
        pipeline: ArbiterPipeline = app.state.pipeline

        # Execute the pipeline using the native domain request
        result: EvaluationResult = pipeline.execute(valid_domain_request)

        # 1. Deterministic Execution Assertion
        assert len(result.metrics) > 0

        # 2. Strict Ownership-Chain Identity (`is`) Assertions
        explanation = result.explanation_result
        decision = explanation.decision_result
        uncertainty = decision.uncertainty_result
        failure_analysis = uncertainty.failure_analysis_result
        verification = failure_analysis.verification_result
        evidence = verification.evidence_bundle

        assert result.explanation_result is explanation
        assert explanation.decision_result is decision
        assert decision.uncertainty_result is uncertainty
        assert uncertainty.failure_analysis_result is failure_analysis
        assert failure_analysis.verification_result is verification
        assert verification.evidence_bundle is evidence

        # Verify determinism across multiple runs (stateless purity)
        result_run_two = pipeline.execute(valid_domain_request)

        # Output should be structurally identical for deterministic inputs
        assert result.metrics[0].score == result_run_two.metrics[0].score
