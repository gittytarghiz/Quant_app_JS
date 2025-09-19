from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from starlette.responses import JSONResponse
import traceback

def setup_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ValueError)
    async def handle_value_error(_: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)}
        )

    @app.exception_handler(ImportError)
    async def handle_import_error(_: Request, exc: ImportError):
        return JSONResponse(
            status_code=500,
            content={"detail": f"Missing dependency: {exc}"}
        )

    @app.exception_handler(RuntimeError)
    async def handle_runtime_error(_: Request, exc: RuntimeError):
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc)}
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception):
        if isinstance(exc, HTTPException):
            raise exc
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Internal error: {exc.__class__.__name__}: {exc}",
                "trace": traceback.format_exc()
            }
        )