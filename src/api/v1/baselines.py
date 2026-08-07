"""FastAPI Personalized Baselines Router (API v1)."""

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_baseline_service, get_repository
from src.models import AthleteBaselineRead
from src.repositories.base import BaseAthleteRepository
from src.services.baseline_service import BaselineService
from src.utils.exceptions import InsufficientDataError

router = APIRouter(prefix="/baselines", tags=["Personalized Baselines"])


@router.get("/{athlete_id}", response_model=AthleteBaselineRead)
def get_athlete_baseline(
    athlete_id: str,
    window_days: int = 30,
    repo: BaseAthleteRepository = Depends(get_repository),
    baseline_service: BaselineService = Depends(get_baseline_service),
):
    """Computes and returns personalized rolling 30/60-day baseline statistics for an athlete."""
    history = repo.get_telemetry_history(athlete_id)
    try:
        baseline = baseline_service.compute_baseline(athlete_id, history, window_days=window_days)
        repo.save_baseline(baseline)
        return baseline
    except InsufficientDataError as e:
        raise HTTPException(status_code=400, detail=str(e.message)) from e
