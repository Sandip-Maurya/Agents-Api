
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

class ExceptionAndLoggingHandler:
    """
    Registers global exception handlers and
    request/response logging middleware on a FastAPI app.
    """

    @staticmethod
    def register(app: FastAPI) -> None:
        # --- Validation errors (422) ---
        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            logger.error(
                "Validation error",
                extra={
                    "path": request.url.path,
                    "errors": exc.errors()
                }
            )
            return JSONResponse(
                status_code=422,
                content={"detail": exc.errors()},
            )

        # --- HTTPExceptions (custom 4xx/5xx) ---
        @app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            logger.warning(
                "HTTP exception",
                extra={
                    "path": request.url.path,
                    "status_code": exc.status_code,
                    "detail": exc.detail
                }
            )
            return JSONResponse(
                status_code=exc.status_code,
                content={"message": exc.detail},
            )

        # --- Catch-all for uncaught exceptions ---
        @app.middleware("http")
        async def catch_exceptions_middleware(request: Request, call_next):
            try:
                response = await call_next(request)
            except Exception as exc:
                # Log stacktrace
                logger.exception("Unhandled exception during request")
                return JSONResponse(
                    status_code=500,
                    content={"message": "Internal server error"},
                )
            return response
