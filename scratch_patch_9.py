with open("src/core/bootstrap.py", "r", encoding="utf-8") as f:
    code = f.read()

code = code.replace("DefaultPipelineLifecycleManager", "PipelineLifecycleManager")
code = code.replace("DefaultPipelineHealthChecker", "PipelineHealthChecker")
code = code.replace("DefaultPipelineReadinessEvaluator", "PipelineReadinessEvaluator")
code = code.replace("DefaultOperationalSnapshotBuilder", "OperationalSnapshotBuilder")
code = code.replace("def _get_records() -> list:", "def _get_records() -> list:\n            return []\n        pipeline.operations = PipelineOperationsController(  # type: ignore")
code = code.replace("def _get_records() -> list:", "def _get_records() -> list[Any]:\n            from typing import Any")

with open("src/core/bootstrap.py", "w", encoding="utf-8") as f:
    f.write(code)
