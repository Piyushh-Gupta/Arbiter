with open("src/api/main.py", "r", encoding="utf-8") as f:
    code = f.read()

import_statement = "from src.core.bootstrap import (\n    build_pipeline,\n    build_resilience_controller,\n    build_resilience_registry,\n    build_telemetry_engine,\n    initialize_application,\n    build_services,\n    build_contract_registry,\n    build_contract_engine\n)"
import_old = "from src.core.bootstrap import (\n    build_pipeline,\n    build_resilience_controller,\n    build_resilience_registry,\n    build_telemetry_engine,\n    initialize_application,\n    build_services,\n)"
code = code.replace(import_old, import_statement)

lifespan_code = '''
        app.state.service_registry = build_services(config, pipeline)
        
        contract_registry = build_contract_registry(config)
        app.state.contract_engine = build_contract_engine(config, contract_registry)
'''
code = code.replace("        app.state.service_registry = build_services(config, pipeline)", lifespan_code)
code = code.replace("    app.state.service_registry = None", "    app.state.contract_engine = None\n    app.state.service_registry = None")

with open("src/api/main.py", "w", encoding="utf-8") as f:
    f.write(code)
