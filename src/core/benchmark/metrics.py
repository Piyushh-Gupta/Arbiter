"""Stateless concrete metric calculators for evaluation."""

import math
from typing import Any, Sequence

from src.core.benchmark.base import BaseMetricCalculator
from src.core.benchmark.benchmark_models import BenchmarkMetricType, MetricResult
from src.core.verification.verification_models import (
    VerificationResult,
    VerificationVerdict,
)


def _extract_labels_and_confs(
    predictions: Sequence[Any],
    ground_truths: Sequence[Any],
) -> tuple[list[str], list[str], list[float]]:
    """Helper to convert predictions and ground truths to string lists and confidences."""
    y_pred = []
    y_true = []
    confidences = []

    for pred in predictions:
        if isinstance(pred, VerificationResult):
            y_pred.append(pred.verdict.value)
            confidences.append(pred.confidence if pred.confidence is not None else 0.0)
        elif isinstance(pred, dict):
            y_pred.append(pred.get("verdict", "INSUFFICIENT"))
            confidences.append(pred.get("confidence", 0.0))
        else:
            y_pred.append(str(pred))
            confidences.append(0.0)

    for gt in ground_truths:
        if isinstance(gt, VerificationVerdict):
            y_true.append(gt.value)
        elif isinstance(gt, dict):
            y_true.append(gt.get("verdict", "INSUFFICIENT"))
        else:
            y_true.append(str(gt))

    return y_pred, y_true, confidences


class AccuracyCalculator(BaseMetricCalculator):
    """Calculates evaluation accuracy."""

    def compute(
        self,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> MetricResult:
        if not predictions:
            return MetricResult(metric_type=BenchmarkMetricType.ACCURACY, value=0.0)

        y_pred, y_true, _ = _extract_labels_and_confs(predictions, ground_truths)
        correct = sum(1 for p, t in zip(y_pred, y_true) if p == t)
        val = correct / len(predictions)
        return MetricResult(metric_type=BenchmarkMetricType.ACCURACY, value=val)


class PrecisionCalculator(BaseMetricCalculator):
    """Calculates macro precision."""

    def compute(
        self,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> MetricResult:
        if not predictions:
            return MetricResult(metric_type=BenchmarkMetricType.PRECISION, value=0.0)

        y_pred, y_true, _ = _extract_labels_and_confs(predictions, ground_truths)
        classes = set(y_true) | set(y_pred)
        precisions = []

        for c in classes:
            tp = sum(1 for p, t in zip(y_pred, y_true) if p == c and t == c)
            fp = sum(1 for p, t in zip(y_pred, y_true) if p == c and t != c)
            denom = tp + fp
            prec = (tp / denom) if denom > 0 else 0.0
            precisions.append(prec)

        val = sum(precisions) / len(classes) if classes else 0.0
        return MetricResult(metric_type=BenchmarkMetricType.PRECISION, value=val)


class RecallCalculator(BaseMetricCalculator):
    """Calculates macro recall."""

    def compute(
        self,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> MetricResult:
        if not predictions:
            return MetricResult(metric_type=BenchmarkMetricType.RECALL, value=0.0)

        y_pred, y_true, _ = _extract_labels_and_confs(predictions, ground_truths)
        classes = set(y_true) | set(y_pred)
        recalls = []

        for c in classes:
            tp = sum(1 for p, t in zip(y_pred, y_true) if p == c and t == c)
            fn = sum(1 for p, t in zip(y_pred, y_true) if p != c and t == c)
            denom = tp + fn
            rec = (tp / denom) if denom > 0 else 0.0
            recalls.append(rec)

        val = sum(recalls) / len(classes) if classes else 0.0
        return MetricResult(metric_type=BenchmarkMetricType.RECALL, value=val)


class F1Calculator(BaseMetricCalculator):
    """Calculates macro F1 score."""

    def compute(
        self,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> MetricResult:
        if not predictions:
            return MetricResult(metric_type=BenchmarkMetricType.F1, value=0.0)

        y_pred, y_true, _ = _extract_labels_and_confs(predictions, ground_truths)
        classes = set(y_true) | set(y_pred)
        f1_list = []

        for c in classes:
            tp = sum(1 for p, t in zip(y_pred, y_true) if p == c and t == c)
            fp = sum(1 for p, t in zip(y_pred, y_true) if p == c and t != c)
            fn = sum(1 for p, t in zip(y_pred, y_true) if p != c and t == c)

            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

            denom = prec + rec
            f1 = (2.0 * prec * rec / denom) if denom > 0.0 else 0.0
            f1_list.append(f1)

        val = sum(f1_list) / len(classes) if classes else 0.0
        return MetricResult(metric_type=BenchmarkMetricType.F1, value=val)


class MacroF1Calculator(BaseMetricCalculator):
    """Calculates macro F1 score (alias for F1)."""

    def compute(
        self,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> MetricResult:
        res = F1Calculator().compute(predictions, ground_truths)
        return MetricResult(metric_type=BenchmarkMetricType.MACRO_F1, value=res.value)


class MicroF1Calculator(BaseMetricCalculator):
    """Calculates micro F1 score."""

    def compute(
        self,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> MetricResult:
        if not predictions:
            return MetricResult(metric_type=BenchmarkMetricType.MICRO_F1, value=0.0)

        y_pred, y_true, _ = _extract_labels_and_confs(predictions, ground_truths)
        classes = set(y_true) | set(y_pred)

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for c in classes:
            total_tp += sum(1 for p, t in zip(y_pred, y_true) if p == c and t == c)
            total_fp += sum(1 for p, t in zip(y_pred, y_true) if p == c and t != c)
            total_fn += sum(1 for p, t in zip(y_pred, y_true) if p != c and t == c)

        denom = total_tp + 0.5 * (total_fp + total_fn)
        val = (total_tp / denom) if denom > 0 else 0.0
        return MetricResult(metric_type=BenchmarkMetricType.MICRO_F1, value=val)


class ECECalculator(BaseMetricCalculator):
    """Calculates Expected Calibration Error."""

    def compute(
        self,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> MetricResult:
        if not predictions:
            return MetricResult(metric_type=BenchmarkMetricType.ECE, value=0.0)

        y_pred, y_true, confs = _extract_labels_and_confs(predictions, ground_truths)
        n = len(predictions)
        bins = 10
        ece = 0.0

        for i in range(bins):
            bin_lower = i / bins
            bin_upper = (i + 1) / bins

            bin_indices = []
            for idx, c in enumerate(confs):
                if i == bins - 1:
                    if bin_lower <= c <= bin_upper:
                        bin_indices.append(idx)
                else:
                    if bin_lower <= c < bin_upper:
                        bin_indices.append(idx)

            if not bin_indices:
                continue

            bin_acc = sum(1 for idx in bin_indices if y_pred[idx] == y_true[idx]) / len(
                bin_indices
            )
            bin_conf = sum(confs[idx] for idx in bin_indices) / len(bin_indices)

            ece += (len(bin_indices) / n) * abs(bin_acc - bin_conf)

        return MetricResult(metric_type=BenchmarkMetricType.ECE, value=ece)


class MCECalculator(BaseMetricCalculator):
    """Calculates Maximum Calibration Error."""

    def compute(
        self,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> MetricResult:
        if not predictions:
            return MetricResult(metric_type=BenchmarkMetricType.MCE, value=0.0)

        y_pred, y_true, confs = _extract_labels_and_confs(predictions, ground_truths)
        bins = 10
        mce = 0.0

        for i in range(bins):
            bin_lower = i / bins
            bin_upper = (i + 1) / bins

            bin_indices = []
            for idx, c in enumerate(confs):
                if i == bins - 1:
                    if bin_lower <= c <= bin_upper:
                        bin_indices.append(idx)
                else:
                    if bin_lower <= c < bin_upper:
                        bin_indices.append(idx)

            if not bin_indices:
                continue

            bin_acc = sum(1 for idx in bin_indices if y_pred[idx] == y_true[idx]) / len(
                bin_indices
            )
            bin_conf = sum(confs[idx] for idx in bin_indices) / len(bin_indices)

            mce = max(mce, abs(bin_acc - bin_conf))

        return MetricResult(metric_type=BenchmarkMetricType.MCE, value=mce)


class BrierScoreCalculator(BaseMetricCalculator):
    """Calculates Brier Score (mean squared error of predicted winning probability)."""

    def compute(
        self,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> MetricResult:
        if not predictions:
            return MetricResult(metric_type=BenchmarkMetricType.BRIER_SCORE, value=0.0)

        y_pred, y_true, confs = _extract_labels_and_confs(predictions, ground_truths)
        brier_sum = 0.0
        for p, t, c in zip(y_pred, y_true, confs):
            p_correct = c if p == t else 0.0
            brier_sum += (p_correct - 1.0) ** 2

        val = brier_sum / len(predictions)
        return MetricResult(metric_type=BenchmarkMetricType.BRIER_SCORE, value=val)


class NegativeLogLikelihoodCalculator(BaseMetricCalculator):
    """Calculates Negative Log Likelihood."""

    def compute(
        self,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> MetricResult:
        if not predictions:
            return MetricResult(
                metric_type=BenchmarkMetricType.NEGATIVE_LOG_LIKELIHOOD, value=0.0
            )

        y_pred, y_true, confs = _extract_labels_and_confs(predictions, ground_truths)
        nll_sum = 0.0
        for p, t, c in zip(y_pred, y_true, confs):
            p_correct = c if p == t else 1e-15
            p_correct = max(1e-15, min(1.0, p_correct))
            nll_sum += -math.log(p_correct)

        val = nll_sum / len(predictions)
        return MetricResult(
            metric_type=BenchmarkMetricType.NEGATIVE_LOG_LIKELIHOOD, value=val
        )


class MeanLatencyCalculator(BaseMetricCalculator):
    """Calculates Mean Latency."""

    def compute(
        self,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> MetricResult:
        # expects predictions to be a sequence of floats representing latencies
        if not predictions:
            return MetricResult(metric_type=BenchmarkMetricType.MEAN_LATENCY, value=0.0)
        val = sum(float(x) for x in predictions) / len(predictions)
        return MetricResult(metric_type=BenchmarkMetricType.MEAN_LATENCY, value=val)


class P95LatencyCalculator(BaseMetricCalculator):
    """Calculates P95 Latency."""

    def compute(
        self,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> MetricResult:
        if not predictions:
            return MetricResult(metric_type=BenchmarkMetricType.P95_LATENCY, value=0.0)
        sorted_preds = sorted(float(x) for x in predictions)
        idx = max(0, min(len(sorted_preds) - 1, int(len(sorted_preds) * 0.95)))
        return MetricResult(
            metric_type=BenchmarkMetricType.P95_LATENCY, value=sorted_preds[idx]
        )


class P99LatencyCalculator(BaseMetricCalculator):
    """Calculates P99 Latency."""

    def compute(
        self,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> MetricResult:
        if not predictions:
            return MetricResult(metric_type=BenchmarkMetricType.P99_LATENCY, value=0.0)
        sorted_preds = sorted(float(x) for x in predictions)
        idx = max(0, min(len(sorted_preds) - 1, int(len(sorted_preds) * 0.99)))
        return MetricResult(
            metric_type=BenchmarkMetricType.P99_LATENCY, value=sorted_preds[idx]
        )


class ThroughputCalculator(BaseMetricCalculator):
    """Calculates throughput (samples/second)."""

    def compute(
        self,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> MetricResult:
        # expects predictions to be a sequence of floats representing latencies
        if not predictions:
            return MetricResult(metric_type=BenchmarkMetricType.THROUGHPUT, value=0.0)
        total_time = sum(float(x) for x in predictions)
        val = len(predictions) / total_time if total_time > 0.0 else 0.0
        return MetricResult(metric_type=BenchmarkMetricType.THROUGHPUT, value=val)


class AbstentionRateCalculator(BaseMetricCalculator):
    """Calculates abstention rate (fraction of INSUFFICIENT verdicts)."""

    def compute(
        self,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> MetricResult:
        if not predictions:
            return MetricResult(
                metric_type=BenchmarkMetricType.ABSTENTION_RATE, value=0.0
            )

        y_pred, _, _ = _extract_labels_and_confs(predictions, ground_truths)
        val = sum(1 for p in y_pred if p == "INSUFFICIENT") / len(predictions)
        return MetricResult(metric_type=BenchmarkMetricType.ABSTENTION_RATE, value=val)


class LowConfidenceRateCalculator(BaseMetricCalculator):
    """Calculates low confidence rate (fraction below threshold, default 0.5)."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def compute(
        self,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> MetricResult:
        if not predictions:
            return MetricResult(
                metric_type=BenchmarkMetricType.LOW_CONFIDENCE_RATE, value=0.0
            )

        _, _, confs = _extract_labels_and_confs(predictions, ground_truths)
        val = sum(1 for c in confs if c < self.threshold) / len(predictions)
        return MetricResult(
            metric_type=BenchmarkMetricType.LOW_CONFIDENCE_RATE, value=val
        )


class ConflictRateCalculator(BaseMetricCalculator):
    """Calculates conflict rate (fraction of samples with high conflict severity, default 0.3)."""

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold

    def compute(
        self,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> MetricResult:
        if not predictions:
            return MetricResult(
                metric_type=BenchmarkMetricType.CONFLICT_RATE, value=0.0
            )

        conflict_count = 0
        for pred in predictions:
            if (
                isinstance(pred, VerificationResult)
                and pred.conflict_analysis is not None
            ):
                if pred.conflict_analysis.conflict_severity > self.threshold:
                    conflict_count += 1

        val = conflict_count / len(predictions)
        return MetricResult(metric_type=BenchmarkMetricType.CONFLICT_RATE, value=val)
