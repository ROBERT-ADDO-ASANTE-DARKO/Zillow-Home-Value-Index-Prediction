"""
Accra Home Price Index – FastAPI backend.

Features
--------
- Auth0 JWT authentication on all /api/* routes
- Redis sliding-window rate limiting (per user sub / IP)
- CORS support for Flutter web and native clients
- Health-check endpoint (unauthenticated)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .cache import close_redis, init_redis
from .config import get_settings
from .routers import ahpi, districts, forecasts, prime


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_redis()
    yield
    await close_redis()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

settings = get_settings()

app = FastAPI(
    title="Accra Home Price Index API",
    description=(
        "REST API for the Accra Home Price Index (AHPI) – "
        "aggregate, district-level, and prime-area price indices "
        "with Prophet-based forecasts (bear / base / bull scenarios)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: propagate Auth0 sub to request.state for rate limiting
# ---------------------------------------------------------------------------

@app.middleware("http")
async def extract_user_sub(request: Request, call_next):
    """
    Attempt to decode the JWT sub without fully verifying it here
    (the route-level dependency does the secure verification).
    This is only used to key the rate-limiter by user identity.
    """
    try:
        from jose import jwt as _jwt

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
            unverified = _jwt.get_unverified_claims(token)
            request.state.user_sub = unverified.get("sub")
        else:
            request.state.user_sub = None
    except Exception:
        request.state.user_sub = None

    return await call_next(request)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

API_PREFIX = "/api/v1"

app.include_router(ahpi.router, prefix=API_PREFIX)
app.include_router(districts.router, prefix=API_PREFIX)
app.include_router(prime.router, prefix=API_PREFIX)
app.include_router(forecasts.router, prefix=API_PREFIX)


# ---------------------------------------------------------------------------
# Health check (public)
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
async def health():
    """Unauthenticated health-check used by Render's health-check probe."""
    return {"status": "ok", "service": "ahpi-api"}


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
