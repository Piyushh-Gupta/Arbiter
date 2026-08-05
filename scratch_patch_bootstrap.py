with open("src/core/bootstrap.py", "r", encoding="utf-8") as f:
    code = f.read()

import_statement = """
from src.api.services.factory import ServiceFactory
from src.api.services.profiles import ServiceProfile
from src.api.services.registry import ServiceProfileRegistry, ServiceRegistry
from src.core.exceptions import APIServiceConfigurationError
"""

if "from src.api.services.factory import ServiceFactory" not in code[:1000]:
    code = code.replace("from src.core.config import Settings", import_statement + "\nfrom src.core.config import Settings", 1)

with open("src/core/bootstrap.py", "w", encoding="utf-8") as f:
    f.write(code)
