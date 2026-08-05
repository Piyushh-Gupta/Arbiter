with open("src/api/main.py", "r", encoding="utf-8") as f:
    code = f.read()

code = code.replace("raise ExceptionTranslator.translate(exc)", '''translated = ExceptionTranslator.translate(exc)
        return JSONResponse(
            status_code=translated.status_code,
            content={"detail": translated.detail},
        )''')

with open("src/api/main.py", "w", encoding="utf-8") as f:
    f.write(code)

import os
from glob import glob

for file in glob("tests/**/*.py", recursive=True):
    with open(file, "r", encoding="utf-8") as f:
        code = f.read()
    
    modified = False
    if "assert response.json() == {\"status\": \"alive\"}" in code:
        code = code.replace("assert response.json() == {\"status\": \"alive\"}", "data = response.json()\n    assert data[\"status\"] == \"alive\"\n    assert \"correlation_id\" in data")
        modified = True
        
    if "assert response.json() == {\"status\": \"ready\"}" in code:
        code = code.replace("assert response.json() == {\"status\": \"ready\"}", "data = response.json()\n    assert data[\"status\"] == \"ready\"\n    assert \"correlation_id\" in data")
        modified = True
        
    if "assert response_200.json() == {\"status\": \"ready\"}" in code:
        code = code.replace("assert response_200.json() == {\"status\": \"ready\"}", "data = response_200.json()\n        assert data[\"status\"] == \"ready\"\n        assert \"correlation_id\" in data")
        modified = True
        
    if "assert response_503.json() == {\"status\": \"not_ready\"}" in code:
        code = code.replace("assert response_503.json() == {\"status\": \"not_ready\"}", "data = response_503.json()\n    assert data[\"status\"] == \"not_ready\"\n    assert \"correlation_id\" in data")
        modified = True

    if "app.state.pipeline = MockPipeline()" in code:
        code = code.replace("app.state.pipeline = MockPipeline()", "pipeline = MockPipeline()\n    from src.api.services.factory import ServiceFactory\n    app.state.pipeline = pipeline\n    app.state.service_registry = ServiceFactory.build_registry(pipeline)")
        modified = True
        
    if "app.state.pipeline = MockPipeline(should_fail=True)" in code:
        code = code.replace("app.state.pipeline = MockPipeline(should_fail=True)", "pipeline = MockPipeline(should_fail=True)\n    from src.api.services.factory import ServiceFactory\n    app.state.pipeline = pipeline\n    app.state.service_registry = ServiceFactory.build_registry(pipeline)")
        modified = True

    if modified:
        with open(file, "w", encoding="utf-8") as f:
            f.write(code)
