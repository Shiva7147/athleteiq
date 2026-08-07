"""FastAPI AI RAG Decision Support & Audit Router (API v1)."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import (
    get_decision_audit_service,
    get_repository,
    get_risk_service,
    get_workload_service,
)
from src.models import DecisionLogRead, DecisionQueryRequest
from src.repositories.base import BaseAthleteRepository
from src.services.decision_audit_service import DecisionAuditService
from src.services.risk_service import RiskService
from src.services.workload_service import WorkloadService
from src.utils.exceptions import InsufficientDataError

router = APIRouter(prefix="/decisions", tags=["AI Decision Support & Audit Logs"])


@router.post("/query", response_model=DecisionLogRead, status_code=status.HTTP_201_CREATED)
def query_decision_support(
    request: DecisionQueryRequest,
    repo: BaseAthleteRepository = Depends(get_repository),
    workload_service: WorkloadService = Depends(get_workload_service),
    risk_service: RiskService = Depends(get_risk_service),
    audit_service: DecisionAuditService = Depends(get_decision_audit_service),
):
    """Submits a natural language query, generates an evidence-backed recommendation with alternative options,

    and persists to database audit log.
    """
    athlete = repo.get_athlete(request.athlete_id)
    if not athlete:
        raise HTTPException(status_code=404, detail=f"Athlete '{request.athlete_id}' not found.")

    history = repo.get_telemetry_history(request.athlete_id)
    try:
        workload = workload_service.compute_summary(history)
        risk = risk_service.evaluate_risk(history, target_sleep_hours=athlete.target_sleep_hours)
    except InsufficientDataError as e:
        raise HTTPException(status_code=400, detail=str(e.message)) from e

    return audit_service.generate_and_audit_decision(
        athlete_id=request.athlete_id,
        query_text=request.query_text,
        workload=workload,
        risk=risk,
        repo=repo,
    )


@router.get("/history/{athlete_id}", response_model=List[DecisionLogRead])
def get_decision_history(
    athlete_id: str,
    repo: BaseAthleteRepository = Depends(get_repository),
):
    """Retrieves decision audit logs for an athlete."""
    return repo.get_decision_history(athlete_id)
