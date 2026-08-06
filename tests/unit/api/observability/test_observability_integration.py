"""Integration tests for API Observability."""

from fastapi.testclient import TestClient

from src.api.main import app
from src.api.observability.telemetry_models import ObservationEvent


def test_api_lifespan_mounts_monitoring_service() -> None:
    with TestClient(app) as client:
        # State mounted during lifespan
        assert hasattr(app.state, "monitoring_service")
        assert app.state.monitoring_service is not None
        assert hasattr(app.state, "monitoring_registry")
        assert app.state.monitoring_registry is not None

        # Test request
        response = client.get("/health/live")
        assert response.status_code == 200

        # Simulate observation processing
        obs = ObservationEvent(
            event_id="test-e",
            timestamp_ns=1000,
            correlation_id="corr-test",
            route_path="/health",
            http_method="GET",
            status_code=200,
            duration_ns=5000,
        )
        app.state.monitoring_service.process_observation(obs)

        metrics = app.state.monitoring_service.get_current_metrics()
        assert metrics.total_requests >= 1
