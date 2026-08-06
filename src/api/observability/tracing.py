"""TracingProvider implementation for API Observability."""

import uuid

from src.api.observability.base import TimeProvider
from src.api.observability.telemetry_models import ObservationEvent, RequestTrace


class TracingProvider:
    """Creates request traces and propagates trace metadata.

    Responsibilities:
    - Produce immutable RequestTrace payloads
    - Extract and inject trace context headers
    - Tie spans to correlation context
    """

    def __init__(self, time_provider: TimeProvider) -> None:
        self._time_provider = time_provider

    def create_trace(self, event: ObservationEvent) -> RequestTrace:
        """Creates an immutable RequestTrace for an ObservationEvent."""
        span_id = f"span-{uuid.uuid4().hex[:12]}"
        raw_trace_id = (
            event.metadata.get("trace_id")
            if event.metadata and "trace_id" in event.metadata
            else None
        )
        trace_id = (
            str(raw_trace_id)
            if raw_trace_id is not None
            else f"trace-{uuid.uuid4().hex[:16]}"
        )
        raw_parent_id = (
            event.metadata.get("parent_span_id")
            if event.metadata and "parent_span_id" in event.metadata
            else None
        )
        parent_span_id = str(raw_parent_id) if raw_parent_id is not None else None

        start_time_ns = event.timestamp_ns
        end_time_ns = start_time_ns + event.duration_ns

        return RequestTrace(
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            span_id=span_id,
            correlation_id=event.correlation_id,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            attributes={
                "http.method": event.http_method,
                "http.route": event.route_path,
                "http.status_code": event.status_code,
                "request_bytes": event.request_bytes,
                "response_bytes": event.response_bytes,
            },
        )

    def extract_trace_context(self, headers: dict[str, str]) -> dict[str, str]:
        """Extracts trace headers from an incoming request header map."""
        normalized = {k.lower(): v for k, v in headers.items()}
        extracted = {}
        if "x-trace-id" in normalized:
            extracted["trace_id"] = normalized["x-trace-id"]
        if "x-parent-span-id" in normalized:
            extracted["parent_span_id"] = normalized["x-parent-span-id"]
        elif "traceparent" in normalized:
            parts = normalized["traceparent"].split("-")
            if len(parts) >= 3:
                extracted["trace_id"] = parts[1]
                extracted["parent_span_id"] = parts[2]
        return extracted

    def inject_trace_context(self, trace: RequestTrace) -> dict[str, str]:
        """Injects trace metadata into HTTP response headers."""
        return {
            "x-trace-id": trace.trace_id,
            "x-span-id": trace.span_id,
            "x-correlation-id": trace.correlation_id,
        }
