"""Production retrieval optimization controller and execution coordinator."""

import time
from typing import Any

from src.core.exceptions import (
    OptimizationConfigurationError,
    OptimizationExecutionError,
    OptimizationTimeoutError,
)
from src.core.reranking.reranking_models import RerankingDefinition
from src.core.retrieval.optimization.concurrency import ConcurrencyLimiter
from src.core.retrieval.optimization.optimization_models import (
    ExecutionPolicy,
    RetrievalExecutionMetrics,
)
from src.core.retrieval.optimization.telemetry import TelemetryCollector
from src.core.retrieval.retrieval_models import EvidenceBundle, RetrievalDefinition


class OptimizationController:
    """
    Stateless controller coordinating optimized retrieval execution policies,
    concurrency bounds, and observational metrics collection.
    """

    def __init__(
        self,
        execution_policy: ExecutionPolicy,
        concurrency_limiter: ConcurrencyLimiter,
        telemetry_collector: TelemetryCollector | None = None,
    ) -> None:
        if not isinstance(execution_policy, ExecutionPolicy):
            raise OptimizationConfigurationError("Requires valid ExecutionPolicy.")
        self._policy = execution_policy
        self._limiter = concurrency_limiter
        self._telemetry = telemetry_collector

    @property
    def execution_policy(self) -> ExecutionPolicy:
        return self._policy

    @property
    def concurrency_limiter(self) -> ConcurrencyLimiter:
        return self._limiter

    @property
    def telemetry_collector(self) -> TelemetryCollector | None:
        return self._telemetry

    def execute(
        self,
        claim: str,
        retriever: Any,
        definition: RetrievalDefinition | Any | None = None,
        reranker: Any | None = None,
        reranking_definition: RerankingDefinition | Any | None = None,
    ) -> tuple[EvidenceBundle, RetrievalExecutionMetrics]:
        t0_total = time.perf_counter()

        # Acquire concurrency slot bounded by timeout
        acquired = self._limiter.acquire(timeout_ms=self._policy.request_timeout_ms)
        if not acquired:
            raise OptimizationTimeoutError(
                f"Execution timed out waiting for concurrency slot after {self._policy.request_timeout_ms}ms."
            )

        try:
            # Stage 1: Retrieval
            t0_ret = time.perf_counter()
            if hasattr(retriever, "retrieve"):
                retriever_func = getattr(retriever, "retrieve")
                if definition is not None:
                    try:
                        bundle = retriever_func(
                            claim,
                            retrieval_definition=definition,
                            reranking_definition=reranking_definition,
                        )
                    except TypeError:
                        bundle = retriever_func(claim, definition)
                else:
                    bundle = retriever_func(claim)
            elif callable(retriever):
                bundle = retriever(claim)
            else:
                raise OptimizationExecutionError(
                    "Retriever must implement retrieve or be callable."
                )
            t1_ret = time.perf_counter()
            ret_ms = (t1_ret - t0_ret) * 1000.0

            if not isinstance(bundle, EvidenceBundle):
                raise OptimizationExecutionError(
                    f"Retriever returned invalid type {type(bundle).__name__}, expected EvidenceBundle."
                )

            # Stage 2: Reranking (if separate reranker provided)
            t0_rerank = time.perf_counter()
            if reranker is not None and hasattr(reranker, "rerank"):
                rerank_func = getattr(reranker, "rerank")
                if reranking_definition is not None:
                    bundle = rerank_func(claim, bundle, reranking_definition)
                else:
                    bundle = rerank_func(claim, bundle)
            t1_rerank = time.perf_counter()
            rerank_ms = (t1_rerank - t0_rerank) * 1000.0

            t1_total = time.perf_counter()
            total_ms = (t1_total - t0_total) * 1000.0

            if total_ms > self._policy.request_timeout_ms:
                raise OptimizationTimeoutError(
                    f"Execution exceeded timeout policy ({total_ms:.2f}ms > {self._policy.request_timeout_ms}ms)."
                )

            is_cache_hit = getattr(bundle.metadata, "strategy_id", "") == "cache"

            metrics = RetrievalExecutionMetrics(
                retrieval_latency_ms=ret_ms,
                reranking_latency_ms=rerank_ms,
                cache_latency_ms=ret_ms if is_cache_hit else 0.0,
                document_lookup_latency_ms=0.0,
                total_latency_ms=total_ms,
                candidate_count=len(bundle.passages),
                passage_count=len(bundle.passages),
                cache_hit=is_cache_hit,
            )

            if self._telemetry is not None:
                self._telemetry.record_execution(metrics)

            return bundle, metrics
        finally:
            self._limiter.release()
