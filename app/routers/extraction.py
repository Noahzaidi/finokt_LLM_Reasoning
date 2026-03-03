"""
Extraction router — HTTP endpoints for the OCR Reasoning Engine.

Endpoints:
- POST /api/v1/extract — LLM-based field extraction (JWT required)
- GET  /api/v1/health  — Service health check (no auth)
- GET  /api/v1/model-info — Model version and capabilities (JWT required)
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from app.middleware.auth import verify_token
from app.models.request import ExtractionRequest
from app.models.response import ExtractionResponse, ErrorResponse

logger = logging.getLogger("fai_llm.router")

router = APIRouter(prefix="/api/v1")


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        500: {"model": ErrorResponse, "description": "Extraction failed"},
        503: {"model": ErrorResponse, "description": "vLLM unavailable"},
    },
)
async def extract_fields(
    request: ExtractionRequest,
    app_request: Request,
    _token: str = Depends(verify_token),
) -> ExtractionResponse:
    """
    Extract missing fields from OCR tokens using LLM reasoning.

    Accepts structured OCR context and returns reasoned field extractions
    with confidence scores and reasoning traces.
    """
    llm_service = app_request.app.state.llm_service

    try:
        response = await llm_service.extract(request)
        return response
    except Exception as e:
        logger.error(
            "Extraction failed: request_id=%s error=%s",
            request.request_id,
            type(e).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="LLM extraction failed. vLLM may be unavailable.",
        )


@router.get("/health")
async def health_check(app_request: Request) -> dict:
    """
    Health check endpoint — no authentication required.

    Returns service status, GPU availability, and model readiness.
    """
    llm_service = app_request.app.state.llm_service
    vllm_health = await llm_service.check_health()

    return {
        "status": "healthy" if vllm_health["vllm_status"] == "healthy" else "degraded",
        "service": "ocr-reasoning-engine",
        "vllm": vllm_health,
    }


@router.get("/model-info")
async def model_info(
    app_request: Request,
    _token: str = Depends(verify_token),
) -> dict:
    """
    Return model version and capabilities. JWT required.
    """
    llm_service = app_request.app.state.llm_service

    return {
        "model_id": llm_service.model_path,
        "model_version": llm_service.model_version,
        "max_tokens": llm_service.max_tokens,
        "temperature": llm_service.temperature,
        "guided_json": True,
        "capabilities": [
            "field_extraction",
            "document_classification",
            "ocr_error_correction",
        ],
    }
