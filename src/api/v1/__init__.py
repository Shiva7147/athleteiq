"""API v1 routers package."""

from fastapi import APIRouter
from src.api.v1.analytics import router as analytics_router
from src.api.v1.athletes import router as athletes_router
from src.api.v1.decisions import router as decisions_router
from src.api.v1.risk import router as risk_router
from src.api.v1.telemetry import router as telemetry_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(athletes_router)
api_v1_router.include_router(telemetry_router)
api_v1_router.include_router(analytics_router)
api_v1_router.include_router(risk_router)
api_v1_router.include_router(decisions_router)

__all__ = ["api_v1_router"]
