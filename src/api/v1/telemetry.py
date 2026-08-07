"""FastAPI Telemetry Router (API v1)."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_repository
from src.models import TelemetryCreate, TelemetryRead
from src.repositories.base import BaseAthleteRepository

router = APIRouter(prefix="/telemetry", tags=["Telemetry"])


@router.post("", response_model=TelemetryRead, status_code=status.HTTP_201_CREATED)
def log_telemetry(
    telemetry: TelemetryCreate,
    repo: BaseAthleteRepository = Depends(get_repository),
):
    """Logs daily wearable telemetry entry."""
    athlete = repo.get_athlete(telemetry.athlete_id)
    if not athlete:
        raise HTTPException(status_code=404, detail=f"Athlete with ID '{telemetry.athlete_id}' not found.")
    return repo.log_telemetry(telemetry)


@router.get("/{athlete_id}", response_model=List[TelemetryRead])
def get_telemetry_history(
    athlete_id: str,
    repo: BaseAthleteRepository = Depends(get_repository),
):
    """Retrieves chronologically sorted telemetry history for an athlete."""
    return repo.get_telemetry_history(athlete_id)
