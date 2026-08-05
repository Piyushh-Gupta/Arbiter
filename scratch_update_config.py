with open("src/core/config.py", "r", encoding="utf-8") as f:
    code = f.read()

settings_class = """class APIServiceSettings(BaseModel):
    \"\"\"Settings for the API Service Layer.\"\"\"
    active_profile_id: str = Field(default="default_api_service")
    require_correlation_id: bool = Field(default=True)
    timeout_seconds: float = Field(default=30.0)

"""

if "class APIServiceSettings" not in code:
    code = code.replace("class Settings(BaseSettings):", settings_class + "class Settings(BaseSettings):")
    code = code.replace("pipeline_operations: PipelineOperationsSettings = Field(default_factory=PipelineOperationsSettings)", "pipeline_operations: PipelineOperationsSettings = Field(default_factory=PipelineOperationsSettings)\n    api_services: APIServiceSettings = Field(default_factory=APIServiceSettings)")
    with open("src/core/config.py", "w", encoding="utf-8") as f:
        f.write(code)
with open('src/core/config.py', 'r', encoding='utf-8') as f:
    content = f.read()

settings_str = '''
class ApiContractSettings(BaseModel):
    \"\"\"Settings for the API Contract Layer.\"\"\"

    active_profile_id: str = Field(default=\"default_api_contract\")
    require_correlation_id: bool = Field(default=True)
    strict_validation: bool = Field(default=True)

class Settings(BaseSettings):
'''

content = content.replace('class Settings(BaseSettings):', settings_str)
content = content.replace('api_services: \"APIServiceSettings\" = Field(', '''api_contracts: \"ApiContractSettings\" = Field(
        default_factory=lambda: ApiContractSettings()
    )
    api_services: \"APIServiceSettings\" = Field(''')

with open('src/core/config.py', 'w', encoding='utf-8') as f:
    f.write(content)
