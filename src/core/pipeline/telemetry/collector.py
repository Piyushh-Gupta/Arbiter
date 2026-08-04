"""In-memory thread-safe collector implementation."""

import threading
from datetime import datetime, timezone

from src.core.pipeline.telemetry.base import BaseTelemetryCollector
from src.core.pipeline.telemetry.telemetry_models import (
    PipelineStageAggregation,
    PipelineStageTelemetryRecord,
    PipelineTelemetryEvent,
    PipelineTelemetrySnapshot,
)


class InMemoryTelemetryCollector(BaseTelemetryCollector):
    """Thread-safe in-memory telemetry collector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[PipelineTelemetryEvent] = []

    def record(self, event: PipelineTelemetryEvent) -> None:
        """Records a single telemetry event thread-safely."""
        with self._lock:
            self._events.append(event)

    def reset(self) -> None:
        """Clears all accumulated telemetry data thread-safely."""
        with self._lock:
            self._events.clear()

    def snapshot(self) -> PipelineTelemetrySnapshot:
        """Produces an immutable aggregated snapshot of all recorded events thread-safely."""
        with self._lock:
            events = list(self._events)

        pipeline_id = events[0].pipeline_id if events else "default_pipeline"
        total_executions = len(events)
        successful_executions = sum(1 for e in events if e.success)
        failed_executions = total_executions - successful_executions
        overall_success_rate = (
            successful_executions / total_executions if total_executions > 0 else 0.0
        )

        total_latencies = [e.total_latency_ms for e in events]
        sorted_total_latencies = sorted(total_latencies)

        def get_percentile(sorted_list: list[float], pct: float) -> float:
            if not sorted_list:
                return 0.0
            idx = int(pct * len(sorted_list))
            idx = max(0, min(idx, len(sorted_list) - 1))
            return sorted_list[idx]

        mean_total_latency = (
            sum(total_latencies) / total_executions if total_executions > 0 else 0.0
        )
        p50_total = get_percentile(sorted_total_latencies, 0.50)
        p90_total = get_percentile(sorted_total_latencies, 0.90)
        p99_total = get_percentile(sorted_total_latencies, 0.99)

        stage_groups: dict[tuple[str, str], list[PipelineStageTelemetryRecord]] = {}
        for event in events:
            for record in event.stage_records:
                key = (record.stage_id, record.profile_id)
                if key not in stage_groups:
                    stage_groups[key] = []
                stage_groups[key].append(record)

        stage_aggregations = []
        for (stage_id, profile_id), records in stage_groups.items():
            stg_count = len(records)
            stg_success = sum(1 for r in records if r.success)
            stg_failure = stg_count - stg_success
            stg_latencies = [r.latency_ms for r in records]
            stg_sorted_latencies = sorted(stg_latencies)

            stg_mean = sum(stg_latencies) / stg_count if stg_count > 0 else 0.0
            stg_p50 = get_percentile(stg_sorted_latencies, 0.50)
            stg_p90 = get_percentile(stg_sorted_latencies, 0.90)
            stg_p99 = get_percentile(stg_sorted_latencies, 0.99)
            stg_rate = stg_success / stg_count if stg_count > 0 else 0.0

            stage_aggregations.append(
                PipelineStageAggregation(
                    stage_id=stage_id,
                    profile_id=profile_id,
                    execution_count=stg_count,
                    success_count=stg_success,
                    failure_count=stg_failure,
                    mean_latency_ms=stg_mean,
                    p50_latency_ms=stg_p50,
                    p90_latency_ms=stg_p90,
                    p99_latency_ms=stg_p99,
                    success_rate=stg_rate,
                )
            )

        stage_aggregations.sort(key=lambda x: x.stage_id)

        return PipelineTelemetrySnapshot(
            pipeline_id=pipeline_id,
            total_executions=total_executions,
            successful_executions=successful_executions,
            failed_executions=failed_executions,
            mean_total_latency_ms=mean_total_latency,
            p50_total_latency_ms=p50_total,
            p90_total_latency_ms=p90_total,
            p99_total_latency_ms=p99_total,
            overall_success_rate=overall_success_rate,
            stage_aggregations=tuple(stage_aggregations),
            snapshot_timestamp=datetime.now(timezone.utc),
        )
