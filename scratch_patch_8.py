import os

# Fix 1: src/api/services/health_service.py
with open("src/api/services/health_service.py", "r", encoding="utf-8") as f:
    code = f.read()

code = code.replace("snapshot.readiness_status ==", "snapshot.overall_readiness ==")

with open("src/api/services/health_service.py", "w", encoding="utf-8") as f:
    f.write(code)

# Fix 2: tests/unit/api/services/test_evaluation_service.py
with open("tests/unit/api/services/test_evaluation_service.py", "r", encoding="utf-8") as f:
    code = f.read()

code = code.replace("evaluation_result=MagicMock(metrics=(mock_metric,))", "metrics=(mock_metric,)")

with open("tests/unit/api/services/test_evaluation_service.py", "w", encoding="utf-8") as f:
    f.write(code)

# Fix 3: tests/unit/test_verification_hardening.py
with open("tests/unit/test_verification_hardening.py", "r", encoding="utf-8") as f:
    code = f.read()

code = code.replace("assert response_live.json() == {\"status\": \"alive\"}", "data = response_live.json()\n        assert data[\"status\"] == \"alive\"\n        assert \"correlation_id\" in data")
code = code.replace("assert response_ready.json() == {\"status\": \"not_ready\"}", "data = response_ready.json()\n        assert data[\"status\"] == \"not_ready\"\n        assert \"correlation_id\" in data")

with open("tests/unit/test_verification_hardening.py", "w", encoding="utf-8") as f:
    f.write(code)
