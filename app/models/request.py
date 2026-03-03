"""
Pydantic models for extraction request payloads.

Defines the structured input contract for the POST /api/v1/extract endpoint.
All fields mirror the API contract defined in the project specification.
"""

from pydantic import BaseModel, Field
from typing import Optional


class BBox(BaseModel):
    """Bounding box coordinates for an OCR token."""
    x: int
    y: int
    w: int
    h: int


class OcrToken(BaseModel):
    """A single OCR token extracted by the upstream OCR system."""
    text: str
    bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0)
    page: int = Field(ge=1)
    line: int = Field(ge=1)
    block: int = Field(ge=1)


class FieldExemplar(BaseModel):
    """A past analyst correction used as a few-shot example for the LLM."""
    field_key: str
    original_value: str
    corrected_value: str


class NormalizationRule(BaseModel):
    """Output format rule for a specific field."""
    field_key: str
    output_format: str


class RagContext(BaseModel):
    """RAG context assembled by the upstream OCR service."""
    field_exemplars: list[FieldExemplar] = Field(default_factory=list)
    normalization_rules: list[NormalizationRule] = Field(default_factory=list)


class ExtractionRequest(BaseModel):
    """
    Full extraction request from the upstream OCR service.

    Sent when the OCR system has low confidence on one or more fields
    and needs LLM-based reasoning to resolve them.
    """
    request_id: str
    tenant_id: str
    document_id: str
    document_type_guess: str
    classification_confidence: float = Field(ge=0.0, le=1.0)
    ocr_tokens: list[OcrToken]
    required_fields: list[str]
    missing_fields: list[str]
    locked_fields: dict[str, str] = Field(default_factory=dict)
    rag_context: Optional[RagContext] = None
    page_images: Optional[list[str]] = None  # Reserved for V2 (vision)
