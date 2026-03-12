"""
Redis client initialisation.

The client is created once at startup and closed at shutdown via the
FastAPI lifespan context manager in main.py.
"""
import redis.asyncio as aioredis

from .config import get_settings

# Module-level reference; populated during app startup
_redis_client: aioredis.Redis | None = None


async def init_redis() -> None:
    global _redis_client
    settings = get_settings()
    _redis_client = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
    )
    # Smoke-test the connection
    await _redis_client.ping()


async def close_redis() -> None:
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None


def get_redis() -> aioredis.Redis:
    if _redis_client is None:
        raise RuntimeError("Redis client is not initialised.")
    return _redis_client
