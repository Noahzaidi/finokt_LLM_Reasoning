"""
RAG service — structural retrieval context assembly.

In the current architecture, RAG context (field exemplars and normalization
rules) is assembled by the upstream OCR service and passed in the request.
This module provides utilities for future enhancements where the LLM
service may need to enrich or validate the RAG context.
"""

import logging

from app.models.request import RagContext

logger = logging.getLogger("fai_llm.rag_service")


def validate_rag_context(rag_context: RagContext | None) -> RagContext | None:
    """
    Validate and sanitize RAG context before prompt assembly.

    Currently passes through as-is since the upstream OCR service
    is responsible for assembling correct context. Future versions
    may add deduplication, relevance filtering, or limit enforcement.
    """
    if rag_context is None:
        return None

    logger.debug(
        "RAG context: %d exemplars, %d rules",
        len(rag_context.field_exemplars),
        len(rag_context.normalization_rules),
    )

    return rag_context
