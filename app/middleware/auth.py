"""
JWT authentication middleware.

Validates Bearer tokens on protected endpoints. Supports two modes:
1. JWT tokens signed with JWT_SECRET (for future external callers)
2. Raw INTERNAL_SERVICE_TOKEN match (for service-to-service calls from OCR)

The health endpoint is excluded from authentication.
"""

import os
import logging
from typing import Optional

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger("fai_llm.auth")

_security = HTTPBearer()


def _get_jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET", "")
    if not secret:
        raise RuntimeError("JWT_SECRET environment variable is not set")
    return secret


def _get_service_token() -> str:
    token = os.getenv("INTERNAL_SERVICE_TOKEN", "")
    if not token:
        raise RuntimeError("INTERNAL_SERVICE_TOKEN environment variable is not set")
    return token


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> str:
    """
    FastAPI dependency that validates the Bearer token.

    Tries two strategies:
    1. Direct match against INTERNAL_SERVICE_TOKEN (service-to-service)
    2. JWT decode using JWT_SECRET (future external callers)

    Returns the validated token string on success.
    Raises HTTP 401 on failure.
    """
    token = credentials.credentials

    # Strategy 1: Direct service token match
    service_token = _get_service_token()
    if token == service_token:
        return token

    # Strategy 2: JWT decode
    try:
        from jose import jwt, JWTError

        jwt_secret = _get_jwt_secret()
        payload = jwt.decode(token, jwt_secret, algorithms=["HS256"])
        return token
    except ImportError:
        logger.error("python-jose not installed, JWT decode unavailable")
    except JWTError as e:
        logger.warning("JWT validation failed: %s", type(e).__name__)
    except Exception:
        logger.warning("Token validation failed")

    raise HTTPException(
        status_code=401,
        detail="Invalid or expired authentication token",
    )
