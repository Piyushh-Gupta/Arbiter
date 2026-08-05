import os
from glob import glob

for file in glob("tests/**/*.py", recursive=True):
    with open(file, "r", encoding="utf-8") as f:
        code = f.read()
    
    modified = False

    if "metadata={\"environment\": \"test\", \"version\": \"1.0.0\"}" in code:
        code = code.replace("PipelineReadinessStatus\n", "PipelineReadinessStatus, PipelineOperationalMetadata\n")
        code = code.replace("metadata={\"environment\": \"test\", \"version\": \"1.0.0\"}", "metadata=PipelineOperationalMetadata(environment=\"test\", version=\"1.0.0\")")
        modified = True
        
    if "ServiceFactory.build_registry(pipeline)" in code:
        code = code.replace("ServiceFactory.build_registry(pipeline)", "ServiceFactory.build_registry(pipeline)  # type: ignore")
        modified = True

    if modified:
        with open(file, "w", encoding="utf-8") as f:
            f.write(code)
