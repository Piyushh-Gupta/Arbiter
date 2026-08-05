with open("src/core/bootstrap.py", "r", encoding="utf-8") as f:
    code = f.read()

import_statement = "from src.api.services.registry import ServiceRegistry\nfrom src.api.contracts.versioning import ApiContractDefinition, ApiContractProfile, ApiContractRegistry, ApiVersionId\nfrom src.api.contracts.engine import ApiContractEngine\n"
code = code.replace("from src.api.services.registry import ServiceRegistry", import_statement)

registry_builder = '''
def build_contract_registry(config: Any) -> ApiContractRegistry:
    """Builds the API Contract registry."""
    definition = ApiContractDefinition(
        supported_versions=(ApiVersionId.V1, ApiVersionId.V2),
        require_correlation_id=getattr(getattr(config, "api_contracts", None), "require_correlation_id", True),
        strict_validation=getattr(getattr(config, "api_contracts", None), "strict_validation", True),
    )
    profile = ApiContractProfile(
        profile_id=getattr(getattr(config, "api_contracts", None), "active_profile_id", "default_api_contract"),
        definition=definition,
    )
    return ApiContractRegistry(profiles=[profile])

def build_contract_engine(config: Any, registry: ApiContractRegistry) -> ApiContractEngine:
    """Builds the API Contract Engine."""
    active_profile_id = getattr(getattr(config, "api_contracts", None), "active_profile_id", "default_api_contract")
    return ApiContractEngine(registry=registry, active_profile_id=active_profile_id)
'''
code = code + "\n" + registry_builder

with open("src/core/bootstrap.py", "w", encoding="utf-8") as f:
    f.write(code)
