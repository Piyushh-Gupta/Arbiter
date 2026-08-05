with open("src/core/exceptions.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(len(lines)):
    if 'class ApiContractConfigurationError(ConfigurationError):' in lines[i]:
        lines[i+1] = '    """Raised when the API contract subsystem is misconfigured."""\n'
    elif 'class ApiContractValidationError(ArbiterError):' in lines[i]:
        lines[i+1] = '    """Raised when an API request or response fails structural validation."""\n'
    elif 'class DuplicateApiContractProfileError(ArbiterError):' in lines[i]:
        lines[i+1] = '    """Raised when a DuplicateApiContractProfileError detects a duplicate profile_id."""\n'
    elif 'class ApiContractProfileNotFoundError(ArbiterError):' in lines[i]:
        lines[i+1] = '    """Raised when an ApiContractProfile cannot be resolved from the registry."""\n'

with open("src/core/exceptions.py", "w", encoding="utf-8") as f:
    f.writelines(lines)
