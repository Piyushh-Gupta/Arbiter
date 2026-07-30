"""Concrete execution engine for offline retrieval benchmarking."""

import time
from typing import Any

from src.core.exceptions import BenchmarkConfigurationError, BenchmarkExecutionError
from src.core.reranking.base import BaseReranker
from src.core.retrieval.base import BaseRetriever
from src.core.retrieval.benchmarking.base import BaseRetrievalBenchmark
from src.core.retrieval.benchmarking.benchmark_models import (
    AggregateMetrics,
    BenchmarkDataset,
    BenchmarkDefinition,
    BenchmarkEnvironmentMetadata,
    BenchmarkReport,
    LatencySummary,
    PerQueryMetric,
)
from src.core.retrieval.benchmarking.metrics import MetricRegistry
from src.core.retrieval.retrieval_models import EvidenceBundle


class RetrievalEvaluator(BaseRetrievalBenchmark):
    """
    Stateless concrete execution strategy for offline retrieval benchmarking.
    Resolves metrics exclusively through an injected MetricRegistry and measures monotonic latency.
    """

    def __init__(self, metric_registry: MetricRegistry) -> None:
        if not isinstance(metric_registry, MetricRegistry):
            raise BenchmarkConfigurationError(
                f"RetrievalEvaluator requires MetricRegistry, got {type(metric_registry).__name__}"
            )
        self._metric_registry = metric_registry

    @property
    def metric_registry(self) -> MetricRegistry:
        return self._metric_registry

    def validate_compatibility(self, definition: BenchmarkDefinition) -> None:
        if not isinstance(definition, BenchmarkDefinition):
            raise BenchmarkConfigurationError(
                f"RetrievalEvaluator requires BenchmarkDefinition, got {type(definition).__name__}"
            )

        if definition.top_k <= 0:
            raise BenchmarkConfigurationError("top_k must be positive.")

        for metric_name in definition.metrics:
            try:
                self._metric_registry.resolve(metric_name)
            except Exception as e:
                raise BenchmarkConfigurationError(
                    f"Metric '{metric_name}' required by definition is not registered: {e}"
                ) from e

    def evaluate_dataset(
        self,
        dataset: BenchmarkDataset,
        retriever: BaseRetriever | Any,
        definition: BenchmarkDefinition,
        reranker: BaseReranker | Any | None = None,
        retrieval_definition: Any | None = None,
        reranking_definition: Any | None = None,
        environment_metadata: BenchmarkEnvironmentMetadata | None = None,
    ) -> BenchmarkReport:
        self.validate_compatibility(definition)

        if not isinstance(dataset, BenchmarkDataset):
            raise BenchmarkConfigurationError("Expected BenchmarkDataset.")

        if not dataset.queries:
            raise BenchmarkExecutionError("BenchmarkDataset contains no queries.")

        per_query_results: list[PerQueryMetric] = []
        latencies_ms: list[float] = []
        cache_hits: int = 0

        # Monotonic time execution loop
        for q in dataset.queries:
            t0 = time.perf_counter()
            try:
                if hasattr(retriever, "retrieve"):
                    retriever_func = getattr(retriever, "retrieve")
                    if retrieval_definition is not None:
                        try:
                            bundle = retriever_func(
                                q.query_text,
                                retrieval_definition=retrieval_definition,
                                reranking_definition=reranking_definition,
                            )
                        except TypeError:
                            bundle = retriever_func(q.query_text, retrieval_definition)
                    else:
                        bundle = retriever_func(q.query_text)
                elif callable(retriever):
                    bundle = retriever(q.query_text)
                else:
                    raise BenchmarkExecutionError(
                        "Retriever must implement retrieve or be callable."
                    )

                if reranker is not None and hasattr(reranker, "rerank"):
                    rerank_func = getattr(reranker, "rerank")
                    if reranking_definition is not None:
                        bundle = rerank_func(q.query_text, bundle, reranking_definition)
                    else:
                        bundle = rerank_func(q.query_text, bundle)
            except Exception as e:
                if isinstance(
                    e, (BenchmarkExecutionError, BenchmarkConfigurationError)
                ):
                    raise
                raise BenchmarkExecutionError(
                    f"Failed retrieving evidence for query '{q.query_id}': {e}"
                ) from e
            t1 = time.perf_counter()

            if not isinstance(bundle, EvidenceBundle):
                raise BenchmarkExecutionError(
                    f"Retriever returned invalid type {type(bundle).__name__}, expected EvidenceBundle."
                )

            latency_ms = (t1 - t0) * 1000.0
            latencies_ms.append(latency_ms)

            is_cache_hit = getattr(bundle.metadata, "strategy_id", "") == "cache"
            if is_cache_hit:
                cache_hits += 1

            # Compute metrics resolved dynamically from MetricRegistry
            recall_calc = self._metric_registry.resolve("recall")
            precision_calc = self._metric_registry.resolve("precision")
            mrr_calc = self._metric_registry.resolve("mrr")
            ndcg_calc = self._metric_registry.resolve("ndcg")
            hit_calc = self._metric_registry.resolve("hit_rate")

            recall = recall_calc.calculate(
                bundle.passages,
                q.expected_span_ids,
                q.expected_document_ids,
                definition.top_k,
            )
            precision = precision_calc.calculate(
                bundle.passages,
                q.expected_span_ids,
                q.expected_document_ids,
                definition.top_k,
            )
            mrr = mrr_calc.calculate(
                bundle.passages,
                q.expected_span_ids,
                q.expected_document_ids,
                definition.top_k,
            )
            ndcg = ndcg_calc.calculate(
                bundle.passages,
                q.expected_span_ids,
                q.expected_document_ids,
                definition.top_k,
            )
            hit = (
                hit_calc.calculate(
                    bundle.passages,
                    q.expected_span_ids,
                    q.expected_document_ids,
                    definition.top_k,
                )
                > 0.0
            )

            per_query_results.append(
                PerQueryMetric(
                    query_id=q.query_id,
                    recall=recall,
                    precision=precision,
                    mrr=mrr,
                    ndcg=ndcg,
                    hit=hit,
                    latency_ms=latency_ms,
                    cache_hit=is_cache_hit,
                    metrics={
                        "recall": recall,
                        "precision": precision,
                        "mrr": mrr,
                        "ndcg": ndcg,
                    },
                )
            )

        # Monotonic latency calculations
        latencies_sorted = sorted(latencies_ms)
        n = len(latencies_sorted)
        mean_ms = sum(latencies_sorted) / n
        min_ms = latencies_sorted[0]
        max_ms = latencies_sorted[-1]
        p50_ms = latencies_sorted[int(n * 0.50)]
        p90_ms = latencies_sorted[int(n * 0.90)]
        p99_ms = latencies_sorted[int(n * 0.99)]

        latency_summary = LatencySummary(
            min_ms=min_ms,
            max_ms=max_ms,
            p50_ms=p50_ms,
            p90_ms=p90_ms,
            p99_ms=p99_ms,
            mean_ms=mean_ms,
        )

        mean_recall = sum(pq.recall for pq in per_query_results) / n
        mean_precision = sum(pq.precision for pq in per_query_results) / n
        mean_mrr = sum(pq.mrr for pq in per_query_results) / n
        mean_ndcg = sum(pq.ndcg for pq in per_query_results) / n
        hit_rate = sum(1.0 for pq in per_query_results if pq.hit) / n

        aggregate_metrics = AggregateMetrics(
            mean_recall=mean_recall,
            mean_precision=mean_precision,
            mrr=mean_mrr,
            mean_ndcg=mean_ndcg,
            hit_rate=hit_rate,
            metrics={
                "mean_recall": mean_recall,
                "mean_precision": mean_precision,
                "mrr": mean_mrr,
                "mean_ndcg": mean_ndcg,
                "hit_rate": hit_rate,
            },
        )

        if environment_metadata is None:
            environment_metadata = BenchmarkEnvironmentMetadata(
                retrieval_profile_id="default_retrieval",
                corpus_version="default",
                benchmark_dataset_fingerprint=dataset.dataset_fingerprint,
                execution_device="cpu",
            )

        return BenchmarkReport(
            benchmark_id=dataset.benchmark_id,
            timestamp=time.time(),
            corpus_version=environment_metadata.corpus_version,
            dataset_fingerprint=dataset.dataset_fingerprint,
            environment_metadata=environment_metadata,
            total_queries=n,
            aggregate_metrics=aggregate_metrics,
            latency_summary=latency_summary,
            cache_hit_ratio=float(cache_hits / n),
            per_query_metrics=tuple(per_query_results),
        )
