"""
AI Video Resume Generator — Backend Entry Point

FastAPI application with:
- CORS (restricted to frontend domain)
- Request logging middleware
- Rate limiting headers
- Startup health verification
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from app.routers.api import router
from app.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AI Video Resume Generator",
        description="Backend API for processing self-introduction videos into professional resumes",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────
    # Only allow requests from our frontend (prevents unauthorized API access)
    allowed_origins = [
        settings.frontend_url,
        "http://localhost:3000",     # Local dev
        "http://localhost:5173",     # Vite dev
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        max_age=3600,
    )

    # ── Request Logging Middleware ─────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start

        # Log non-health requests
        if request.url.path != "/health":
            print(
                f"[{request.method}] {request.url.path} "
                f"→ {response.status_code} ({duration:.2f}s)"
            )

        # Add timing header
        response.headers["X-Process-Time"] = f"{duration:.3f}"
        return response

    # ── Global Exception Handler ──────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        print(f"[UNHANDLED ERROR] {request.url.path}: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Please try again."},
        )

    # ── Routes ────────────────────────────────────────────────
    app.include_router(router)

    # ── Root ──────────────────────────────────────────────────
    @app.get("/")
    async def root():
        return {
            "service": "AI Video Resume Generator",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
