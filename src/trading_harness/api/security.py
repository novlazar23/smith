from __future__ import annotations

from fastapi import HTTPException, Query, Request, status

from trading_harness.config import get_settings


def _get_api_key(request: Request, key_header: str, query_param: str) -> str | None:
    """Extract API key from header or query parameter."""
    # Check header first
    key = request.headers.get(key_header)
    if key:
        return key
    # Fall back to query parameter
    return request.query_params.get(query_param)


def _verify_key(provided: str | None, expected: str) -> None:
    """Verify an API key matches the expected value.

    Raises HTTPException 401 if key is missing or invalid.
    """
    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
        )
    if provided != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )


def require_read_key(request: Request) -> None:
    """Dependency: require API key for read endpoints.

    If read_api_key is not set in config, reads are unauthenticated.
    """
    settings = get_settings()
    if not settings.read_api_key:
        return  # No key configured, allow access
    provided = _get_api_key(request, "X-Read-API-Key", "read_api_key")
    _verify_key(provided, settings.read_api_key)


def require_trade_key(request: Request) -> None:
    """Dependency: require API key for trade/execution endpoints.

    If trade_api_key is not set in config, trade endpoints are unauthenticated.
    """
    settings = get_settings()
    if not settings.trade_api_key:
        return  # No key configured, allow access
    provided = _get_api_key(request, "X-Trade-API-Key", "trade_api_key")
    _verify_key(provided, settings.trade_api_key)


# ---------------------------------------------------------------------------
# Legacy/compat aliases
# ---------------------------------------------------------------------------

# These keep the old API key style if someone was using query params directly.
def api_key_query(read_key: str | None = Query(None)) -> str:
    """Legacy: extract read API key from query parameter."""
    return read_key or ""