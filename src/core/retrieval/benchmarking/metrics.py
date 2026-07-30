"""Stateless metric calculators and MetricRegistry for retrieval benchmarking."""

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from src.core.exceptions import DuplicateMetricError, MetricNotFoundError
from src.core.retrieval.benchmarking.base import MetricCalculator
from src.core.retrieval.retrieval_models import EvidencePassage

__all__ = [
    "HitRateCalculator",
    "MRRCalculator",
    "MetricRegistry",
    "NDCGCalculator",
    "PrecisionCalculator",
    "RecallCalculator",
    "compute_hit_rate",
    "compute_mrr",
    "compute_ndcg",
    "compute_precision",
    "compute_recall",
]


def _extract_matched_spans(
    retrieved_passages: Sequence[EvidencePassage],
    expected_span_ids: Sequence[str],
    top_k: int,
) -> tuple[int, int, list[bool]]:
    k_passages = retrieved_passages[:top_k]
    expected_set = set(expected_span_ids)
    if not expected_set:
        return (0, len(k_passages), [False] * len(k_passages))

    hits = [p.span_id in expected_set for p in k_passages]
    matched_count = sum(1 for h in hits if h)
    return (matched_count, len(expected_set), hits)


def compute_recall(
    retrieved_passages: Sequence[EvidencePassage],
    expected_span_ids: Sequence[str],
    expected_doc_ids: Sequence[str] = (),
    top_k: int = 5,
) -> float:
    if not expected_span_ids:
        return 0.0
    matched, total_expected, _ = _extract_matched_spans(
        retrieved_passages, expected_span_ids, top_k
    )
    return float(matched / total_expected) if total_expected > 0 else 0.0


def compute_precision(
    retrieved_passages: Sequence[EvidencePassage],
    expected_span_ids: Sequence[str],
    expected_doc_ids: Sequence[str] = (),
    top_k: int = 5,
) -> float:
    k_passages = retrieved_passages[:top_k]
    if not k_passages:
        return 0.0
    matched, _, _ = _extract_matched_spans(retrieved_passages, expected_span_ids, top_k)
    return float(matched / len(k_passages))


def compute_mrr(
    retrieved_passages: Sequence[EvidencePassage],
    expected_span_ids: Sequence[str],
    expected_doc_ids: Sequence[str] = (),
    top_k: int = 5,
) -> float:
    _, _, hits = _extract_matched_spans(retrieved_passages, expected_span_ids, top_k)
    for idx, hit in enumerate(hits, start=1):
        if hit:
            return float(1.0 / idx)
    return 0.0


def compute_ndcg(
    retrieved_passages: Sequence[EvidencePassage],
    expected_span_ids: Sequence[str],
    expected_doc_ids: Sequence[str] = (),
    top_k: int = 5,
) -> float:
    _, total_expected, hits = _extract_matched_spans(
        retrieved_passages, expected_span_ids, top_k
    )
    if total_expected == 0 or not hits:
        return 0.0

    dcg = sum(
        (1.0 if hit else 0.0) / math.log2(rank + 1)
        for rank, hit in enumerate(hits, start=1)
    )
    idcg = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, min(top_k, total_expected) + 1)
    )
    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_hit_rate(
    retrieved_passages: Sequence[EvidencePassage],
    expected_span_ids: Sequence[str],
    expected_doc_ids: Sequence[str] = (),
    top_k: int = 5,
) -> float:
    matched, _, _ = _extract_matched_spans(retrieved_passages, expected_span_ids, top_k)
    return 1.0 if matched > 0 else 0.0


class RecallCalculator:
    def calculate(
        self,
        retrieved_passages: Sequence[EvidencePassage],
        expected_span_ids: Sequence[str],
        expected_doc_ids: Sequence[str] = (),
        top_k: int = 5,
    ) -> float:
        return compute_recall(
            retrieved_passages, expected_span_ids, expected_doc_ids, top_k
        )


class PrecisionCalculator:
    def calculate(
        self,
        retrieved_passages: Sequence[EvidencePassage],
        expected_span_ids: Sequence[str],
        expected_doc_ids: Sequence[str] = (),
        top_k: int = 5,
    ) -> float:
        return compute_precision(
            retrieved_passages, expected_span_ids, expected_doc_ids, top_k
        )


class MRRCalculator:
    def calculate(
        self,
        retrieved_passages: Sequence[EvidencePassage],
        expected_span_ids: Sequence[str],
        expected_doc_ids: Sequence[str] = (),
        top_k: int = 5,
    ) -> float:
        return compute_mrr(
            retrieved_passages, expected_span_ids, expected_doc_ids, top_k
        )


class NDCGCalculator:
    def calculate(
        self,
        retrieved_passages: Sequence[EvidencePassage],
        expected_span_ids: Sequence[str],
        expected_doc_ids: Sequence[str] = (),
        top_k: int = 5,
    ) -> float:
        return compute_ndcg(
            retrieved_passages, expected_span_ids, expected_doc_ids, top_k
        )


class HitRateCalculator:
    def calculate(
        self,
        retrieved_passages: Sequence[EvidencePassage],
        expected_span_ids: Sequence[str],
        expected_doc_ids: Sequence[str] = (),
        top_k: int = 5,
    ) -> float:
        return compute_hit_rate(
            retrieved_passages, expected_span_ids, expected_doc_ids, top_k
        )


class MetricRegistry(BaseModel):
    """Immutable namespace for securely registering and resolving metric calculators."""

    calculators: dict[str, MetricCalculator] = Field(
        default_factory=dict, description="Named mapping of metric calculators."
    )

    _metric_index: dict[str, MetricCalculator] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_index(self) -> "MetricRegistry":
        index: dict[str, MetricCalculator] = {}
        for name, calc in self.calculators.items():
            norm_name = name.strip().lower()
            if norm_name in index:
                raise DuplicateMetricError(
                    f"Duplicate metric identifier: '{norm_name}'"
                )
            index[norm_name] = calc
        object.__setattr__(self, "_metric_index", index)
        return self

    def resolve(self, metric_name: str) -> MetricCalculator:
        norm_name = metric_name.strip().lower()
        if norm_name not in self._metric_index:
            raise MetricNotFoundError(f"Metric not found: '{metric_name}'")
        return self._metric_index[norm_name]
