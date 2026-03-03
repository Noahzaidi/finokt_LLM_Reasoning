"""
LLM Service — vLLM integration layer.

Handles prompt construction, vLLM API calls, and response parsing.
Uses guided_json to structurally constrain the LLM output to the
ExtractionResponse Pydantic schema, preventing hallucinated fields
and malformed JSON.

PRIVACY: No document content, OCR text, or field values are logged.
"""

import json
import logging
import os
import time
from typing import Optional

import httpx

from app.models.request import ExtractionRequest
from app.models.response import (
    ExtractionResponse,
    ExtractedField,
    LLMExtractionResult,
)

logger = logging.getLogger("fai_llm.llm_service")


SYSTEM_PROMPT = """You are an OCR post-processing reasoning engine for identity documents.
Your task is to extract specific fields from OCR tokens produced by an upstream OCR system.

Rules:
- Only extract fields listed in missing_fields
- Use locked_fields as ground truth — never override them
- Apply normalization rules from the context exactly
- Common OCR misreads: 'O' (letter) is often '0' (zero), 'l' is often '1', 'S' is often '5'
- If a field cannot be found with confidence > 0.7, set its value to null
- Never invent values that don't appear in the tokens
- Output ONLY valid JSON matching the provided schema"""


class LLMService:
    """
    Orchestrates LLM-based field extraction via vLLM.

    Lifecycle:
    - Created once at app startup via lifespan
    - Shared across all requests (stateless — no request data persisted)
    """

    def __init__(self) -> None:
        self.vllm_base_url = os.getenv(
            "FAI_LLM_VLLM_BASE_URL", "http://127.0.0.1:8100/v1"
        )
        self.model_path = os.getenv(
            "FAI_LLM_MODEL", "/opt/fai-llm/models/qwen2.5-7b-awq"
        )
        self.model_version = os.getenv("FAI_LLM_MODEL_VERSION", "qwen2.5-7b-awq")
        self.max_tokens = int(os.getenv("FAI_LLM_MAX_TOKENS", "512"))
        self.temperature = float(os.getenv("FAI_LLM_TEMPERATURE", "0.0"))

        self._client = httpx.AsyncClient(timeout=120.0)
        self._guided_json_schema = LLMExtractionResult.model_json_schema()

        logger.info(
            "LLMService initialized: vllm_url=%s model_version=%s",
            self.vllm_base_url,
            self.model_version,
        )

    async def check_health(self) -> dict:
        """Check vLLM health and return status info."""
        try:
            resp = await self._client.get(f"{self.vllm_base_url}/models")
            resp.raise_for_status()
            models = resp.json()
            return {
                "vllm_status": "healthy",
                "models_loaded": len(models.get("data", [])),
                "model_id": self.model_path,
            }
        except Exception as e:
            return {
                "vllm_status": "unhealthy",
                "error": type(e).__name__,
                "model_id": self.model_path,
            }

    def build_prompt(self, request: ExtractionRequest) -> str:
        """
        Build the user prompt from OCR tokens and context.

        PRIVACY: This prompt is sent to the local vLLM server only.
        It never leaves the machine.
        """
        # Format OCR tokens with spatial context
        formatted_tokens = "\n".join(
            f"  [{t.line}:{t.block}] \"{t.text}\" "
            f"(conf={t.confidence:.2f}, bbox=[{t.bbox.x},{t.bbox.y},{t.bbox.w},{t.bbox.h}])"
            for t in request.ocr_tokens
        )

        # Format RAG context if available
        exemplars_text = "None"
        rules_text = "None"
        if request.rag_context:
            if request.rag_context.field_exemplars:
                exemplars_text = json.dumps(
                    [e.model_dump() for e in request.rag_context.field_exemplars],
                    indent=2,
                    ensure_ascii=False,
                )
            if request.rag_context.normalization_rules:
                rules_text = json.dumps(
                    [r.model_dump() for r in request.rag_context.normalization_rules],
                    indent=2,
                )

        locked_text = json.dumps(request.locked_fields, ensure_ascii=False)

        return (
            f"Document type guess: {request.document_type_guess} "
            f"(confidence: {request.classification_confidence})\n"
            f"Required fields: {json.dumps(request.required_fields)}\n"
            f"Missing fields (need extraction): {json.dumps(request.missing_fields)}\n"
            f"Locked fields (already confirmed): {locked_text}\n\n"
            f"Past analyst corrections for similar documents:\n{exemplars_text}\n\n"
            f"Normalization rules:\n{rules_text}\n\n"
            f"OCR Tokens:\n{formatted_tokens}\n\n"
            f"Extract the missing fields: {json.dumps(request.missing_fields)}"
        )

    async def call_vllm(self, user_prompt: str) -> dict:
        """
        Send a chat completion request to the local vLLM server.

        Uses guided_json to structurally constrain the output.
        Temperature is set to 0.0 for deterministic extraction.
        """
        payload = {
            "model": self.model_path,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "guided_json": self._guided_json_schema,
        }

        resp = await self._client.post(
            f"{self.vllm_base_url}/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()

    def parse_response(
        self, raw_response: dict, request: ExtractionRequest, latency_ms: int
    ) -> ExtractionResponse:
        """
        Parse vLLM response into ExtractionResponse.

        Merges the LLM output with request metadata (IDs, model version, latency).
        """
        content = raw_response["choices"][0]["message"]["content"]
        llm_result = LLMExtractionResult.model_validate_json(content)

        return ExtractionResponse(
            request_id=request.request_id,
            document_id=request.document_id,
            document_type=llm_result.document_type,
            classification_confidence=llm_result.classification_confidence,
            extracted_fields=llm_result.extracted_fields,
            model_version=self.model_version,
            latency_ms=latency_ms,
        )

    async def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        """
        Full extraction pipeline: build prompt → call vLLM → parse response.

        Logs only request_id, tenant_id, document_id, and latency.
        Never logs document content or field values.
        """
        logger.info(
            "Extraction started: request_id=%s tenant_id=%s document_id=%s",
            request.request_id,
            request.tenant_id,
            request.document_id,
        )

        start = time.time()

        user_prompt = self.build_prompt(request)
        raw_response = await self.call_vllm(user_prompt)
        latency_ms = int((time.time() - start) * 1000)
        response = self.parse_response(raw_response, request, latency_ms)

        logger.info(
            "Extraction completed: request_id=%s latency_ms=%d fields_extracted=%d",
            request.request_id,
            latency_ms,
            len(response.extracted_fields),
        )

        return response

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
