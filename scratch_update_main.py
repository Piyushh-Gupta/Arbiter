with open("src/api/main.py", "r", encoding="utf-8") as f:
    code = f.read()

import_stmt = "from src.core.bootstrap import build_services\nfrom src.api.services.exceptions import ExceptionTranslator\n"

if "build_services" not in code:
    code = code.replace("from src.core.bootstrap import (", import_stmt + "from src.core.bootstrap import (")
    code = code.replace("app.state.pipeline = pipeline", "app.state.pipeline = pipeline\n        app.state.service_registry = build_services(config, pipeline)")
    code = code.replace("app.state.pipeline = None", "app.state.service_registry = None\n    app.state.pipeline = None")

    code = code.replace("return JSONResponse(\n            status_code=status.HTTP_400_BAD_REQUEST,\n            content={\"detail\": str(exc)},\n        )", "raise ExceptionTranslator.translate(exc)")
    code = code.replace("return JSONResponse(\n            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,\n            content={\"detail\": \"Internal Server Error\"},\n        )", "raise ExceptionTranslator.translate(exc)")

    with open("src/api/main.py", "w", encoding="utf-8") as f:
        f.write(code)
