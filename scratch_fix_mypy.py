import re

files_to_fix = [
    "src/api/contracts/versioning.py",
    "tests/unit/api/contracts/test_response_models.py",
    "tests/unit/api/contracts/test_request_models.py",
    "tests/unit/api/contracts/test_error_models.py"
]

for file in files_to_fix:
    with open(file, "r", encoding="utf-8") as f:
        code = f.read()
    if file == "src/api/contracts/versioning.py":
        code = code.replace("self._profiles: Mapping[str, ApiContractProfile] = {}", "self._profiles: dict[str, ApiContractProfile] = {}")
        code = code.replace("self._profiles[profile.profile_id] = profile  # type: ignore", "self._profiles[profile.profile_id] = profile")
    else:
        code = code.replace("  # type: ignore", "")
    with open(file, "w", encoding="utf-8") as f:
        f.write(code)
