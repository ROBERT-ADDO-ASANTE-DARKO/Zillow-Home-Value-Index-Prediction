"""
Auth0 JWT validation for FastAPI.

Verifies RS256-signed JWTs issued by Auth0 using JWKS endpoint.
"""
import httpx
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError

from .config import Settings, get_settings

bearer_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# JWKS caching
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_jwks(domain: str) -> dict:
    """Fetch and cache Auth0 JWKS (public keys)."""
    url = f"https://{domain}/.well-known/jwks.json"
    response = httpx.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def _get_signing_key(token: str, domain: str) -> str:
    """Extract the RSA public key that matches the JWT kid header."""
    jwks = _get_jwks(domain)
    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")
    for key in jwks["keys"]:
        if key["kid"] == kid:
            return key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Public key not found in JWKS",
    )


# ---------------------------------------------------------------------------
# Token verification
# ---------------------------------------------------------------------------

def verify_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer_scheme)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    """
    Dependency that validates the Auth0 JWT access token.

    Returns the decoded token payload (claims) on success.
    Raises HTTP 401 on any failure.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        signing_key = _get_signing_key(token, settings.auth0_domain)
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=settings.auth0_algorithms,
            audience=settings.auth0_audience,
            issuer=f"https://{settings.auth0_domain}/",
        )
        return payload

    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Convenience alias for use in route dependencies
AuthUser = Annotated[dict, Depends(verify_token)]
