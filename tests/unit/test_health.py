from fastapi.testclient import TestClient

from src.api.main import app


def test_health_check_live() -> None:
    client = TestClient(app)
    response = client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"
    assert "correlation_id" in data
