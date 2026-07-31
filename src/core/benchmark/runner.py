"""Concrete verification benchmark runner execution logic."""

import hashlib
import json
import time
from typing import Any

from src.core.benchmark.base import (
    BaseBenchmark,
    BaseBenchmarkDataset,
    BaseMetricCalculator,
)
from src.core.benchmark.benchmark_models import (
    BenchmarkDefinition,
    BenchmarkMetricType,
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkTrace,
)
from src.core.benchmark.metrics import (
    AbstentionRateCalculator,
    AccuracyCalculator,
    BrierScoreCalculator,
    ConflictRateCalculator,
    ECECalculator,
    F1Calculator,
    LowConfidenceRateCalculator,
    MacroF1Calculator,
    MCECalculator,
    MeanLatencyCalculator,
    MicroF1Calculator,
    NegativeLogLikelihoodCalculator,
    P95LatencyCalculator,
    P99LatencyCalculator,
    PrecisionCalculator,
    RecallCalculator,
    ThroughputCalculator,
)
from src.core.verification.verification_models import (
    ClaimVerificationInput,
    VerificationVerdict,
)

# Registry mapping BenchmarkMetricType to its stateless calculator instance
METRIC_CALCULATORS: dict[BenchmarkMetricType, BaseMetricCalculator] = {
    BenchmarkMetricType.ACCURACY: AccuracyCalculator(),
    BenchmarkMetricType.PRECISION: PrecisionCalculator(),
    BenchmarkMetricType.RECALL: RecallCalculator(),
    BenchmarkMetricType.F1: F1Calculator(),
    BenchmarkMetricType.MACRO_F1: MacroF1Calculator(),
    BenchmarkMetricType.MICRO_F1: MicroF1Calculator(),
    BenchmarkMetricType.ECE: ECECalculator(),
    BenchmarkMetricType.MCE: MCECalculator(),
    BenchmarkMetricType.BRIER_SCORE: BrierScoreCalculator(),
    BenchmarkMetricType.NEGATIVE_LOG_LIKELIHOOD: NegativeLogLikelihoodCalculator(),
    BenchmarkMetricType.MEAN_LATENCY: MeanLatencyCalculator(),
    BenchmarkMetricType.P95_LATENCY: P95LatencyCalculator(),
    BenchmarkMetricType.P99_LATENCY: P99LatencyCalculator(),
    BenchmarkMetricType.THROUGHPUT: ThroughputCalculator(),
    BenchmarkMetricType.ABSTENTION_RATE: AbstentionRateCalculator(),
    BenchmarkMetricType.LOW_CONFIDENCE_RATE: LowConfidenceRateCalculator(),
    BenchmarkMetricType.CONFLICT_RATE: ConflictRateCalculator(),
}


class VerificationBenchmarkRunner(BaseBenchmark):
    """Orchestrates benchmark dataset loading, verification execution, and metric calculations."""

    def __init__(
        self,
        verification_registry: Any,
        calibration_registry: Any,
        datasets: dict[str, BaseBenchmarkDataset],
    ) -> None:
        self.verification_registry = verification_registry
        self.calibration_registry = calibration_registry
        self.datasets = datasets

    def run_benchmark(
        self,
        definition: BenchmarkDefinition,
    ) -> BenchmarkReport:
        # 1. Resolve dataset
        dataset_id = definition.dataset_identifier
        if dataset_id not in self.datasets:
            raise KeyError(
                f"Dataset '{dataset_id}' not found in registered runner datasets."
            )
        dataset = self.datasets[dataset_id]
        samples = dataset.load_samples()

        # 2. Resolve verification profile
        profile_id = definition.evaluation_profile_id
        verifier_profile = self.verification_registry.resolve(profile_id)
        verifier = verifier_profile.verifier
        ver_def = verifier_profile.definition

        # 3. Resolve default calibration profile
        # Use first profile in calibration registry as default or look up matching
        # For evaluation context calibration, resolve default calibration profile
        calibration_profile = self.calibration_registry.resolve("identity")

        predictions = []
        ground_truths = []
        latencies = []
        execution_seq = []

        # 4. Iterate over dataset samples
        for idx, sample in enumerate(samples):
            claim = sample["claim"]
            bundle = sample["bundle"]
            gt_verdict_str = sample["ground_truth_verdict"]
            gt_verdict = VerificationVerdict(gt_verdict_str)
            sample_id = sample.get("sample_id", f"sample_{idx}")

            execution_seq.append(sample_id)
            ground_truths.append(gt_verdict)

            # Execution timing
            t_start = time.perf_counter()

            # Execute pipeline: Verify -> Aggregate -> Calibrate
            passage_results = verifier.verify_passages(claim, bundle)
            claim_input = ClaimVerificationInput(
                claim=claim, bundle=bundle, definition=ver_def
            )
            from src.core.verification.aggregation import (
                BaseAggregationStrategy,
                MaxConfidenceAggregationStrategy,
            )

            if isinstance(ver_def.aggregation_strategy, BaseAggregationStrategy):
                agg_strat = ver_def.aggregation_strategy
            else:
                agg_strat = MaxConfidenceAggregationStrategy()
            aggregated = agg_strat.aggregate(claim_input, passage_results)
            calibrated = calibration_profile.strategy.calibrate(
                aggregated,
                calibration_profile.definition,
                calibration_profile.uncertainty_estimator,
            )

            t_end = time.perf_counter()
            latencies.append(t_end - t_start)
            predictions.append(calibrated)

        # 5. Compute selected metrics
        metric_results = []
        for m_type in definition.selected_metrics:
            if m_type not in METRIC_CALCULATORS:
                continue
            calculator = METRIC_CALCULATORS[m_type]

            # Latency and throughput calculators consume the latency list
            if m_type in (
                BenchmarkMetricType.MEAN_LATENCY,
                BenchmarkMetricType.P95_LATENCY,
                BenchmarkMetricType.P99_LATENCY,
                BenchmarkMetricType.THROUGHPUT,
            ):
                m_res = calculator.compute(latencies, [])
            else:
                m_res = calculator.compute(predictions, ground_truths)

            metric_results.append(m_res)

        # Compute confusion matrix
        confusion_matrix: dict[str, dict[str, int]] = {
            v.value: {v2.value: 0 for v2 in VerificationVerdict}
            for v in VerificationVerdict
        }
        for p, t in zip(predictions, ground_truths):
            confusion_matrix[t.value][p.verdict.value] += 1

        # Configuration fingerprint
        fingerprint_input = {
            "verifier_model": ver_def.verifier_model,
            "aggregation_strategy": str(ver_def.aggregation_strategy),
            "calibration_strategy": str(calibration_profile.definition.strategy),
        }
        fingerprint = hashlib.md5(
            json.dumps(fingerprint_input, sort_keys=True).encode()
        ).hexdigest()

        # Build Trace
        trace = BenchmarkTrace(
            dataset_version=dataset.dataset_version(),
            execution_sequence=tuple(execution_seq),
            metric_execution_order=definition.selected_metrics,
            configuration_fingerprint=fingerprint,
            execution_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        # Build Result
        result = BenchmarkResult(
            metrics=tuple(metric_results),
            confusion_matrix=confusion_matrix,
            metadata={"sample_count": len(samples)},
        )

        import platform
        import sys

        env = {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": sys.version,
            "machine": platform.machine(),
        }

        return BenchmarkReport(
            benchmark_result=result,
            benchmark_trace=trace,
            execution_environment=env,
            configuration_fingerprint=fingerprint,
        )
