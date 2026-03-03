"""
Pydantic models for extraction response payloads.

Defines the structured output contract for the POST /api/v1/extract endpoint.
The ExtractionResponse schema is also used as the guided_json constraint
for vLLM, ensuring the model can only produce valid, schema-conformant output.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ExtractedField(BaseModel):
    """A single field extracted by the LLM with confidence and reasoning."""
    value: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "llm"
    reasoning: str


class LLMExtractionResult(BaseModel):
    """
    Schema used as the guided_json constraint for vLLM.

    This is the shape the LLM is structurally forced to produce.
    It is then merged with request metadata to form the full ExtractionResponse.
    """
    document_type: str
    classification_confidence: float = Field(ge=0.0, le=1.0)
    extracted_fields: dict[str, ExtractedField]


class ExtractionResponse(BaseModel):
    """
    Full extraction response returned to the upstream OCR service.

    Includes request context (IDs, model version, latency) alongside
    the LLM's structured extraction result.
    """
    request_id: str
    document_id: str
    document_type: str
    classification_confidence: float
    extracted_fields: dict[str, ExtractedField]
    model_version: str
    latency_ms: int


class ErrorResponse(BaseModel):
    """Structured error response — never return unstructured errors."""
    request_id: Optional[str] = None
    error: str
    detail: Optional[str] = None
