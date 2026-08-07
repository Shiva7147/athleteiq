"""FastAPI Injury Risk Assessment Router (API v1)."""

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_repository, get_risk_service
from src.models import InjuryRiskAssessmentRead
from src.repositories.base import BaseAthleteRepository
from src.services.risk_service import RiskService
from src.utils.exceptions import InsufficientDataError

router = APIRouter(prefix="/risk", tags=["Risk Engine"])


@router.get("/{athlete_id}", response_model=InjuryRiskAssessmentRead)
def evaluate_injury_risk(
    athlete_id: str,
    repo: BaseAthleteRepository = Depends(get_repository),
    risk_service: RiskService = Depends(get_risk_service),
):
    """Runs calibrated soft-tissue injury risk assessment."""
    athlete = repo.get_athlete(athlete_id)
    if not athlete:
        raise HTTPException(status_code=404, detail=f"Athlete with ID '{athlete_id}' not found.")

    history = repo.get_telemetry_history(athlete_id)
    try:
        return risk_service.evaluate_risk(history, target_sleep_hours=athlete.target_sleep_hours)
    except InsufficientDataError as e:
        raise HTTPException(status_code=400, detail=str(e.message)) from e
