import os
from glob import glob

for file in glob("tests/**/*.py", recursive=True):
    with open(file, "r", encoding="utf-8") as f:
        code = f.read()
    
    modified = False

    if "metadata=MagicMock()" in code and "PipelineOperationalSnapshot" in code:
        code = code.replace("metadata=MagicMock()", "metadata={\"environment\": \"test\", \"version\": \"1.0.0\"}")
        modified = True

    if modified:
        with open(file, "w", encoding="utf-8") as f:
            f.write(code)
