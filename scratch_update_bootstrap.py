with open("src/core/bootstrap.py", "r", encoding="utf-8") as f:
    code = f.read()

import_statement = """
from src.api.services.factory import ServiceFactory
from src.api.services.profiles import ServiceProfile
from src.api.services.registry import ServiceProfileRegistry, ServiceRegistry
from src.core.exceptions import APIServiceConfigurationError
"""

if "from src.api.services.factory import ServiceFactory" not in code:
    code = code.replace("from src.core.exceptions import (", import_statement + "\nfrom src.core.exceptions import (", 1)

build_services_func = """
def build_services(config: AppConfig, pipeline: Any) -> ServiceRegistry:
    \"\"\"Builds and validates the API Service Layer.\"\"\"
    settings = config.api_services
    
    try:
        profile = ServiceProfile(
            profile_id=settings.active_profile_id,
            require_correlation_id=settings.require_correlation_id,
            timeout_seconds=settings.timeout_seconds,
        )
        registry = ServiceProfileRegistry(profiles=(profile,))
    except Exception as e:
        raise APIServiceConfigurationError(f"Service registry validation failed: {e}") from e

    # Construct the services exclusively through the ServiceFactory
    service_registry = ServiceFactory.build_registry(pipeline)
    return service_registry
"""

if "def build_services" not in code:
    code = code + "\n\n" + build_services_func

with open("src/core/bootstrap.py", "w", encoding="utf-8") as f:
    f.write(code)
