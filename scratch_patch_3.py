with open("src/api/services/evaluation_service.py", "r", encoding="utf-8") as f:
    code = f.read()
code = code.replace("for m in domain_res.evaluation_result.metrics", "for m in domain_res.metrics")
with open("src/api/services/evaluation_service.py", "w", encoding="utf-8") as f:
    f.write(code)

with open("src/core/bootstrap.py", "r", encoding="utf-8") as f:
    code = f.read()
code = code.replace("registry = ServiceProfileRegistry(profiles=(profile,))", "ServiceProfileRegistry(profiles=(profile,))")
with open("src/core/bootstrap.py", "w", encoding="utf-8") as f:
    f.write(code)

with open("tests/unit/api/services/test_service_models.py", "r", encoding="utf-8") as f:
    code = f.read()
code = code.replace("EvaluationRequest(claim=\"test\", pipeline_profile_id=\"default\")  # type: ignore", "EvaluationRequest(claim=\"test\", pipeline_profile_id=\"default\")")
code = code.replace("EvaluationRequest(claim=\"test\", pipeline_profile_id=\"default\")", "EvaluationRequest(claim=\"test\", pipeline_profile_id=\"default\")  # type: ignore")
with open("tests/unit/api/services/test_service_models.py", "w", encoding="utf-8") as f:
    f.write(code)
