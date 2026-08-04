"""Unit and integration tests for M5.2 Pipeline Stage Observability & Telemetry subsystem."""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.main import create_app
from src.core.bootstrap import build_telemetry_engine
from src.core.config import Settings
from src.core.evaluation.evaluation_models import (
    EvaluationMetadata,
    EvaluationMetric,
    EvaluationResult,
)
from src.core.exceptions import (
    DuplicateTelemetryProfileError,
    PipelineStageExecutionError,
    TelemetryConfigurationError,
    TelemetryProfileNotFoundError,
)
from src.core.pipeline.pipeline_models import (
    PipelineExecutionContext,
    PipelineExecutionResult,
    PipelineRuntimeMetadata,
    PipelineStageMetadata,
)
from src.core.pipeline.telemetry import (
    DefaultTelemetryEventFactory,
    InMemoryTelemetryCollector,
    JsonTelemetryExporter,
    JsonTelemetryExporterDefinition,
    LogTelemetryExporter,
    LogTelemetryExporterDefinition,
    PipelineTelemetryDefinition,
    PipelineTelemetryEngine,
    PipelineTelemetryEvent,
    PipelineTelemetrySnapshot,
    TelemetryExporterProfile,
    TelemetryExporterRegistry,
)

# ==========================================
# 1. Model Immutability Tests
# ==========================================


def test_telemetry_models_immutability() -> None:
    # 1. PipelineTelemetryEvent
    event = PipelineTelemetryEvent(
        execution_id="exec_1",
        pipeline_id="pipe_1",
        claim_length=42,
        total_latency_ms=120.5,
        success=True,
        stage_records=(),
        configuration_fingerprint="fingerprint_1",
        execution_environment="test",
        observed_at=datetime.now(timezone.utc),
    )
    with pytest.raises(ValidationError):
        setattr(event, "success", False)

    # 2. PipelineTelemetrySnapshot
    snapshot = PipelineTelemetrySnapshot(
        pipeline_id="pipe_1",
        total_executions=1,
        successful_executions=1,
        failed_executions=0,
        mean_total_latency_ms=120.5,
        p50_total_latency_ms=120.5,
        p90_total_latency_ms=120.5,
        p99_total_latency_ms=120.5,
        overall_success_rate=1.0,
        stage_aggregations=(),
        snapshot_timestamp=datetime.now(timezone.utc),
    )
    with pytest.raises(ValidationError):
        setattr(snapshot, "total_executions", 2)


# ==========================================
# 2. Registry & Definition Validation Tests
# ==========================================


def test_registry_duplicate_detection() -> None:
    log_def = LogTelemetryExporterDefinition(exporter_id="log_exporter")
    exporter = LogTelemetryExporter()
    profile = TelemetryExporterProfile(
        profile_id="dup_profile",
        definition=log_def,
        exporter=exporter,
    )

    # Duplicate profile ID detection in registry
    with pytest.raises(DuplicateTelemetryProfileError):
        TelemetryExporterRegistry(profiles=(profile, profile))


def test_registry_unknown_profile_resolution() -> None:
    log_def = LogTelemetryExporterDefinition(exporter_id="log_exporter")
    exporter = LogTelemetryExporter()
    profile = TelemetryExporterProfile(
        profile_id="valid_profile",
        definition=log_def,
        exporter=exporter,
    )
    registry = TelemetryExporterRegistry(profiles=(profile,))

    assert registry.resolve("valid_profile") is profile

    with pytest.raises(TelemetryProfileNotFoundError):
        registry.resolve("non_existent_profile")


def test_exporter_compatibility_validation() -> None:
    log_def = LogTelemetryExporterDefinition(exporter_id="log_exporter")
    json_exporter = JsonTelemetryExporter()

    # Incompatible pairing (JSON exporter with log definition) must raise TelemetryConfigurationError
    with pytest.raises(TelemetryConfigurationError):
        TelemetryExporterProfile(
            profile_id="bad_pairing",
            definition=log_def,
            exporter=json_exporter,
        )


# ==========================================
# 3. Collector & Snapshot Calculation Tests
# ==========================================


def test_collector_snapshot_metrics() -> None:
    collector = InMemoryTelemetryCollector()

    # Initial snapshot on empty collector
    snap = collector.snapshot()
    assert snap.total_executions == 0
    assert snap.overall_success_rate == 0.0
    assert snap.mean_total_latency_ms == 0.0

    # Record some events with custom latencies
    observed_time = datetime.now(timezone.utc)
    for latency in [10.0, 20.0, 30.0, 40.0, 100.0]:
        event = PipelineTelemetryEvent(
            execution_id=f"exec_{latency}",
            pipeline_id="test_pipeline",
            claim_length=10,
            total_latency_ms=latency,
            success=latency < 100.0,  # one failure (100.0)
            stage_records=(),
            configuration_fingerprint="fingerprint",
            execution_environment="test",
            observed_at=observed_time,
        )
        collector.record(event)

    snap = collector.snapshot()
    assert snap.total_executions == 5
    assert snap.successful_executions == 4
    assert snap.failed_executions == 1
    assert snap.overall_success_rate == 0.8
    assert snap.mean_total_latency_ms == 40.0  # (10+20+30+40+100)/5 = 40.0

    # Deterministic percentiles using sorted list index lookup:
    # sorted total latencies: [10.0, 20.0, 30.0, 40.0, 100.0]
    # P50 (pct=0.5): idx = int(0.5 * 5) = 2 -> 30.0
    # P90 (pct=0.9): idx = int(0.9 * 5) = 4 -> 100.0
    # P99 (pct=0.99): idx = int(0.99 * 5) = 4 -> 100.0
    assert snap.p50_total_latency_ms == 30.0
    assert snap.p90_total_latency_ms == 100.0
    assert snap.p99_total_latency_ms == 100.0

    # Clear collector
    collector.reset()
    assert collector.snapshot().total_executions == 0


def test_collector_thread_safety() -> None:
    collector = InMemoryTelemetryCollector()
    num_threads = 10
    events_per_thread = 100

    def worker(worker_id: int) -> None:
        for i in range(events_per_thread):
            event = PipelineTelemetryEvent(
                execution_id=f"exec_{worker_id}_{i}",
                pipeline_id="thread_test_pipeline",
                claim_length=15,
                total_latency_ms=12.0,
                success=True,
                stage_records=(),
                configuration_fingerprint="fp",
                execution_environment="test",
                observed_at=datetime.now(timezone.utc),
            )
            collector.record(event)

    threads = []
    for t_id in range(num_threads):
        t = threading.Thread(target=worker, args=(t_id,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    snap = collector.snapshot()
    assert snap.total_executions == num_threads * events_per_thread


# ==========================================
# 4. Exporter Strategy Tests
# ==========================================


def test_log_telemetry_exporter(caplog: pytest.LogCaptureFixture) -> None:
    log_def = LogTelemetryExporterDefinition(
        exporter_id="test_log",
        log_level="WARNING",
        include_stage_breakdown=True,
    )
    exporter = LogTelemetryExporter()
    exporter.validate_compatibility(log_def)

    snapshot = PipelineTelemetrySnapshot(
        pipeline_id="log_pipeline",
        total_executions=2,
        successful_executions=2,
        failed_executions=0,
        mean_total_latency_ms=50.0,
        p50_total_latency_ms=50.0,
        p90_total_latency_ms=50.0,
        p99_total_latency_ms=50.0,
        overall_success_rate=1.0,
        stage_aggregations=(),
        snapshot_timestamp=datetime.now(timezone.utc),
    )

    with caplog.at_level(logging.WARNING, logger="arbiter.telemetry"):
        report = exporter.export(snapshot)

    assert report.format == "log"
    assert "log_pipeline" in report.content
    assert "overall_success_rate=100.00%" in report.content
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING


def test_json_telemetry_exporter(tmp_path: Path) -> None:
    output_file = str(tmp_path / "telemetry" / "snapshot.json")
    json_def = JsonTelemetryExporterDefinition(
        exporter_id="test_json",
        output_path=output_file,
        pretty_print=True,
    )
    exporter = JsonTelemetryExporter()
    exporter.validate_compatibility(json_def)

    snapshot = PipelineTelemetrySnapshot(
        pipeline_id="json_pipeline",
        total_executions=1,
        successful_executions=1,
        failed_executions=0,
        mean_total_latency_ms=25.0,
        p50_total_latency_ms=25.0,
        p90_total_latency_ms=25.0,
        p99_total_latency_ms=25.0,
        overall_success_rate=1.0,
        stage_aggregations=(),
        snapshot_timestamp=datetime.now(timezone.utc),
    )

    report = exporter.export(snapshot)
    assert report.format == "json"
    assert os.path.exists(output_file)

    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["pipeline_id"] == "json_pipeline"
    assert data["total_executions"] == 1

    # Test directory creation failure scenario
    bad_def = JsonTelemetryExporterDefinition(
        exporter_id="bad_json",
        output_path=str(
            tmp_path
        ),  # path to a directory, not a file, will fail write check
    )
    bad_exporter = JsonTelemetryExporter()
    bad_exporter.validate_compatibility(bad_def)
    with pytest.raises(PipelineStageExecutionError):
        bad_exporter.export(snapshot)


# ==========================================
# 5. Factory & Engine Coordination Tests
# ==========================================


def test_event_factory_transformation() -> None:
    factory = DefaultTelemetryEventFactory()

    runtime_meta = PipelineRuntimeMetadata(
        pipeline_version="1.0.0",
        configuration_fingerprint="fingerprint_val",
        schema_version="1.0.0",
        execution_environment="test_env",
        execution_timestamp=datetime(2026, 8, 4, 12, 0, 0, tzinfo=timezone.utc),
    )

    exec_ctx = PipelineExecutionContext(
        execution_id="exec_id_val",
        pipeline_id="pipe_id_val",
        claim="This is a test claim.",  # len = 22
        runtime_metadata=runtime_meta,
        stage_metadata=(
            PipelineStageMetadata(
                stage_id="stage_1",
                profile_id="retrieval_profile",
                latency_ms=12.5,
                success=True,
            ),
        ),
        total_latency_ms=100.0,
        success=True,
    )

    from src.core.explainability.explainability_models import (
        ExplanationMetadata,
        ExplanationResult,
        ExplanationSection,
    )

    metric = EvaluationMetric(
        identifier="dummy_metric",
        title="Dummy Metric",
        score=1.0,
        details="Qualitative context",
    )
    eval_meta = EvaluationMetadata(
        strategy_id="dummy_strategy",
    )
    exp_section = ExplanationSection(
        identifier="dummy_section",
        title="Dummy Section",
        content="Dummy Content",
    )
    exp_meta = ExplanationMetadata(
        strategy_id="dummy_strategy",
    )
    explanation = ExplanationResult(
        sections=(exp_section,),
        decision_result=None,
        metadata=exp_meta,
    )

    eval_result = EvaluationResult(
        metrics=(metric,),
        explanation_result=explanation,
        metadata=eval_meta,
    )

    result = PipelineExecutionResult(
        evaluation_result=eval_result,
        execution_context=exec_ctx,
    )

    event = factory.from_result(result)
    assert event.execution_id == "exec_id_val"
    assert event.pipeline_id == "pipe_id_val"
    assert event.claim_length == 21
    assert event.total_latency_ms == 100.0
    assert event.success is True

    assert len(event.stage_records) == 1
    assert event.stage_records[0].stage_id == "stage_1"
    assert event.stage_records[0].profile_id == "retrieval_profile"
    assert event.stage_records[0].latency_ms == 12.5
    assert event.stage_records[0].success is True
    assert event.configuration_fingerprint == "fingerprint_val"
    assert event.execution_environment == "test_env"
    assert event.observed_at == runtime_meta.execution_timestamp


def test_telemetry_engine_swallows_exporter_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Setup a mock exporter that throws an error when export is called
    class FailingExporter:
        def validate_compatibility(
            self, definition: JsonTelemetryExporterDefinition
        ) -> None:
            pass

        def export(self, snapshot: PipelineTelemetrySnapshot) -> None:
            raise RuntimeError("Disk write failure!")

    json_def = JsonTelemetryExporterDefinition(
        exporter_id="failing_exporter",
        output_path="path",
    )
    profile = TelemetryExporterProfile(
        profile_id="default_json_exporter",  # matches engine active exporters
        definition=json_def,
        exporter=FailingExporter(),  # type: ignore
    )
    registry = TelemetryExporterRegistry(profiles=(profile,))

    engine_def = PipelineTelemetryDefinition(
        enabled=True,
        snapshot_on_every_execution=False,
        active_exporter_profile_ids=("default_json_exporter",),
    )

    collector = InMemoryTelemetryCollector()
    event_factory = DefaultTelemetryEventFactory()
    engine = PipelineTelemetryEngine(
        definition=engine_def,
        collector=collector,
        exporter_registry=registry,
        event_factory=event_factory,
    )

    # Setup dummy event
    event = PipelineTelemetryEvent(
        execution_id="exec_1",
        pipeline_id="pipe_1",
        claim_length=5,
        total_latency_ms=10.0,
        success=True,
        stage_records=(),
        configuration_fingerprint="fp",
        execution_environment="test",
        observed_at=datetime.now(timezone.utc),
    )
    collector.record(event)

    # Calling export_snapshot must log and swallow the exception, rather than crashing
    with caplog.at_level(logging.ERROR, logger="arbiter.telemetry"):
        reports = engine.export_snapshot()

    assert len(reports) == 0
    assert len(caplog.records) == 1
    assert "Telemetry export failed for profile 'default_json_exporter'" in caplog.text


# ==========================================
# 6. Bootstrap & API Integration Tests
# ==========================================


def test_bootstrap_telemetry_engine() -> None:
    settings = Settings()
    engine = build_telemetry_engine(settings)
    assert isinstance(engine, PipelineTelemetryEngine)
    assert engine._definition.enabled is True
    assert "default_log_exporter" in engine._definition.active_exporter_profile_ids

    # Fail fast check on configuration error
    settings.pipeline_telemetry.active_exporters = ["non_existent_exporter"]
    with pytest.raises(TelemetryConfigurationError):
        build_telemetry_engine(settings)


def test_api_e2e_telemetry_observation() -> None:
    app = create_app()
    with TestClient(app) as client:
        # Check that telemetry engine is successfully registered on app state
        assert hasattr(app.state, "telemetry_engine")
        engine = app.state.telemetry_engine
        assert isinstance(engine, PipelineTelemetryEngine)

        # Clear any prior events
        engine.reset()

        payload = {
            "claim": "FastAPI is cool.",
            "pipeline_profile_id": "default_pipeline",
        }

        # Send a valid POST /v1/evaluate request
        response = client.post("/v1/evaluate", json=payload)
        assert response.status_code == 200

        # Snapshot should reflect 1 execution
        snapshot = engine._collector.snapshot()
        assert snapshot.total_executions == 1
        assert snapshot.overall_success_rate == 1.0
        assert snapshot.stage_aggregations[0].execution_count == 1
