with open("src/core/exceptions.py", "r", encoding="utf-8") as f:
    code = f.read()

exceptions_class = """class APIServiceConfigurationError(ConfigurationError):
    \"\"\"Raised when the API service layer is misconfigured.\"\"\"
"""

if "class APIServiceConfigurationError" not in code:
    code = code + "\n\n" + exceptions_class
    with open("src/core/exceptions.py", "w", encoding="utf-8") as f:
        f.write(code)
