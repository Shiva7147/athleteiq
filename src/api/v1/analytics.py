"""FastAPI Workload Analytics Router (API v1)."""

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_repository, get_workload_service
from src.models import WorkloadSummaryRead
from src.repositories.base import BaseAthleteRepository
from src.services.workload_service import WorkloadService
from src.utils.exceptions import InsufficientDataError

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/{athlete_id}", response_model=WorkloadSummaryRead)
def get_workload_analytics(
    athlete_id: str,
    repo: BaseAthleteRepository = Depends(get_repository),
    workload_service: WorkloadService = Depends(get_workload_service),
):
    """Computes deterministic workload summary (ACWR, EWMA, Monotony, Strain)."""
    history = repo.get_telemetry_history(athlete_id)
    try:
        return workload_service.compute_summary(history)
    except InsufficientDataError as e:
        raise HTTPException(status_code=400, detail=str(e.message)) from e
