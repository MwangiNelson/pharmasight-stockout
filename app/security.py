"""API-key authentication for the prediction endpoint."""

from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.settings import Settings, get_settings

_api_key = APIKeyHeader(name="X-Api-Key", auto_error=False)


async def require_api_key(key: str | None = Depends(_api_key), settings: Settings = Depends(get_settings)) -> None:
    if not key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing X-Api-Key header")
    if not any(secrets.compare_digest(key, k) for k in settings.keys):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid API key")
