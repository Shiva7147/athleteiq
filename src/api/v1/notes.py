"""FastAPI Qualitative Coach Notes Router (API v1)."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_repository
from src.models import CoachNoteCreate, CoachNoteRead
from src.repositories.base import BaseAthleteRepository

router = APIRouter(prefix="/notes", tags=["Coach Notes"])


@router.post("", response_model=CoachNoteRead, status_code=status.HTTP_201_CREATED)
def create_coach_note(
    note: CoachNoteCreate,
    repo: BaseAthleteRepository = Depends(get_repository),
):
    """Logs a qualitative coach observation note linked to an athlete and telemetry date."""
    athlete = repo.get_athlete(note.athlete_id)
    if not athlete:
        raise HTTPException(status_code=404, detail=f"Athlete with ID '{note.athlete_id}' not found.")
    return repo.create_coach_note(note)


@router.get("/{athlete_id}", response_model=List[CoachNoteRead])
def get_coach_notes(
    athlete_id: str,
    repo: BaseAthleteRepository = Depends(get_repository),
):
    """Retrieves qualitative coach observation notes for an athlete."""
    return repo.get_coach_notes(athlete_id)
