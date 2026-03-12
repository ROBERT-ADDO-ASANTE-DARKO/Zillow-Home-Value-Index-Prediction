"""
Sliding-window rate limiter using Redis.

Each authenticated user is identified by their Auth0 sub claim.
Unauthenticated callers (public endpoints) fall back to client IP.

Algorithm:
  - Maintain a sorted set per key: member = request timestamp, score = timestamp.
  - On each request:
      1. Remove entries older than (now - window).
      2. Count remaining entries.
      3. If count >= limit → 429.
      4. Otherwise add current timestamp and set TTL = window.
"""
import time
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from .cache import get_redis
from .config import Settings, get_settings


async def sliding_window_rate_limit(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """
    FastAPI dependency – raises HTTP 429 when the caller exceeds the
    configured rate limit.

    Attach as a dependency on individual routers or the whole app:

        router = APIRouter(dependencies=[Depends(sliding_window_rate_limit)])
    """
    redis = get_redis()
    limit = settings.rate_limit_requests
    window = settings.rate_limit_window_seconds

    # Prefer authenticated user identity; fall back to IP address
    user_sub: str | None = getattr(request.state, "user_sub", None)
    identifier = user_sub or (request.client.host if request.client else "unknown")
    key = f"rate_limit:{identifier}"

    now = time.time()
    window_start = now - window

    pipe = redis.pipeline()
    pipe.zremrangebyscore(key, "-inf", window_start)   # evict old entries
    pipe.zcard(key)                                     # count remaining
    pipe.zadd(key, {str(now): now})                    # add current request
    pipe.expire(key, window)                            # reset TTL
    results = await pipe.execute()

    request_count: int = results[1]

    if request_count >= limit:
        retry_after = int(window - (now - window_start))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )


RateLimited = Depends(sliding_window_rate_limit)
