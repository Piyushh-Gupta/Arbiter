import os
from glob import glob

for file in glob("tests/**/*.py", recursive=True):
    with open(file, "r", encoding="utf-8") as f:
        code = f.read()
    
    modified = False

    if "class MockPipeline:" in code and "def __init__(self, should_fail: bool = False) -> None:" in code:
        if "self.operations = MagicMock()" not in code:
            code = code.replace("self.should_fail = should_fail", "self.should_fail = should_fail\n        from unittest.mock import MagicMock\n        from src.core.pipeline.operations.operation_models import PipelineOperationalSnapshot, PipelineLifecycleState, PipelineHealthStatus, PipelineReadinessStatus\n        import datetime\n        \n        self.operations = MagicMock()\n        self.operations.get_snapshot.return_value = PipelineOperationalSnapshot(\n            timestamp=datetime.datetime.now(datetime.timezone.utc),\n            lifecycle_state=PipelineLifecycleState.RUNNING,\n            overall_health=PipelineHealthStatus.HEALTHY,\n            overall_readiness=PipelineReadinessStatus.READY,\n            subsystem_records=(),\n            metadata=MagicMock()\n        )")
            modified = True

    if modified:
        with open(file, "w", encoding="utf-8") as f:
            f.write(code)
