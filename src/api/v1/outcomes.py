"""FastAPI Decision Outcomes Router (API v1)."""

from typing import List, Optional
from fastapi import APIRouter, Depends, status

from src.api.dependencies import get_repository
from src.models import ComplianceAnalytics, DecisionOutcomeCreate, DecisionOutcomeRead
from src.repositories.base import BaseAthleteRepository

router = APIRouter(prefix="/outcomes", tags=["Decision Outcomes"])


@router.post("", response_model=DecisionOutcomeRead, status_code=status.HTTP_201_CREATED)
def log_decision_outcome(
    outcome: DecisionOutcomeCreate,
    repo: BaseAthleteRepository = Depends(get_repository),
):
    """Logs coach action taken (Followed / Modified / Ignored) and 7-day injury/performance outcomes."""
    return repo.log_outcome(outcome)


@router.get("", response_model=List[DecisionOutcomeRead])
def get_decision_outcomes(
    athlete_id: Optional[str] = None,
    repo: BaseAthleteRepository = Depends(get_repository),
):
    """Retrieves decision outcomes history."""
    return repo.get_outcomes(athlete_id=athlete_id)


@router.get("/analytics", response_model=ComplianceAnalytics)
def get_compliance_analytics(repo: BaseAthleteRepository = Depends(get_repository)):
    """Computes platform-wide coach compliance rates and injury prevention analytics."""
    return repo.get_compliance_analytics()
