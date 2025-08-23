from __future__ import annotations

import os
from typing import Optional
from fastapi import Header, HTTPException, status


def get_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> Optional[str]:
    expected = os.getenv("API_KEY")
    if expected:
        # Enforce API key if configured
        if not x_api_key or x_api_key != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    return x_api_key


def get_idempotency_key(idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key")) -> Optional[str]:
    return idempotency_key
