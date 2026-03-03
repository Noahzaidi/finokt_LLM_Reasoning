"""
OCR Reasoning Engine — FastAPI application entry point.

Initializes the FastAPI app with:
- Lifespan management (LLMService startup/shutdown)
- Extraction router
- Global exception handler for structured error responses
- Environment loading via python-dotenv

Port architecture:
  8100 → vLLM (localhost only, never exposed)
  8200 → This FastAPI app (0.0.0.0, guarded by firewall)
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.models.response import ErrorResponse
from app.routers.extraction import router as extraction_router
from app.services.llm_service import LLMService

# Load environment variables from .env file
load_dotenv()

# Configure structured logging (no PII)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fai_llm.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    App lifespan: initialize LLMService on startup, clean up on shutdown.

    The LLMService is stored in app.state and shared across all requests.
    It holds no request-specific state (stateless design).
    """
    logger.info("Starting OCR Reasoning Engine...")

    # Initialize LLM service
    llm_service = LLMService()
    app.state.llm_service = llm_service

    # Verify vLLM connectivity
    health = await llm_service.check_health()
    if health["vllm_status"] == "healthy":
        logger.info("vLLM connection verified: model loaded")
    else:
        logger.warning(
            "vLLM not reachable at startup — service will retry on first request"
        )

    yield

    # Shutdown
    logger.info("Shutting down OCR Reasoning Engine...")
    await llm_service.close()


# Create FastAPI app
app = FastAPI(
    title="OCR Reasoning Engine",
    description=(
        "GPU-accelerated LLM microservice for intelligent OCR post-processing. "
        "Extracts fields from OCR tokens using structured LLM reasoning with "
        "guided JSON output."
    ),
    version=os.getenv("FAI_LLM_MODEL_VERSION", "1.0.0"),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Include routers
app.include_router(extraction_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all exception handler.

    Ensures every error returns structured JSON — never an unhandled
    exception reaching the client. Logs the error type only (no PII).
    """
    logger.error("Unhandled exception: %s — %s", type(exc).__name__, str(exc))

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            detail="An unexpected error occurred. Check service logs.",
        ).model_dump(),
    )
