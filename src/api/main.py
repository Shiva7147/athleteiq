"""FastAPI Production Backend Application Entry Point.

Assembles API v1 routers, global exception handlers, CORS middleware,
and database table initialization.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.v1 import api_v1_router
from src.config import settings
from src.database.connection import init_db
from src.utils.exceptions import AthleteIQError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifespan handler."""
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-quality AI Decision Support Platform for Sports Science.",
    lifespan=lifespan,
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AthleteIQError)
async def athleteiq_exception_handler(request: Request, exc: AthleteIQError):
    """Global domain exception handler mapping AthleteIQError to structured HTTP responses."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }


# Include API v1 router aggregator
app.include_router(api_v1_router)
