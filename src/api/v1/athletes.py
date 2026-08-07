"""FastAPI Athletes Router (API v1)."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_repository
from src.models import AthleteCreate, AthleteRead
from src.repositories.base import BaseAthleteRepository

router = APIRouter(prefix="/athletes", tags=["Athletes"])


@router.post("", response_model=AthleteRead, status_code=status.HTTP_201_CREATED)
def create_athlete(
    athlete: AthleteCreate,
    repo: BaseAthleteRepository = Depends(get_repository),
):
    """Creates a new athlete profile."""
    return repo.create_athlete(athlete)


@router.get("", response_model=List[AthleteRead])
def list_athletes(repo: BaseAthleteRepository = Depends(get_repository)):
    """Lists all registered athlete profiles."""
    return repo.list_athletes()


@router.get("/{athlete_id}", response_model=AthleteRead)
def get_athlete(
    athlete_id: str,
    repo: BaseAthleteRepository = Depends(get_repository),
):
    """Retrieves an athlete profile by ID."""
    athlete = repo.get_athlete(athlete_id)
    if not athlete:
        raise HTTPException(status_code=404, detail=f"Athlete with ID '{athlete_id}' not found.")
    return athlete
