"""FastAPI AI RAG Decision Support Router (API v1)."""

from fastapi import APIRouter, Depends, HTTPException

from src.ai.decision_engine import RAGDecisionEngine
from src.api.dependencies import get_rag_engine, get_repository, get_risk_service, get_workload_service
from src.models import DecisionQueryRequest, StructuredDecisionResponse
from src.repositories.base import BaseAthleteRepository
from src.services.risk_service import RiskService
from src.services.workload_service import WorkloadService
from src.utils.exceptions import InsufficientDataError

router = APIRouter(prefix="/decisions", tags=["AI Decision Support"])


@router.post("/query", response_model=StructuredDecisionResponse)
def query_decision_support(
    request: DecisionQueryRequest,
    repo: BaseAthleteRepository = Depends(get_repository),
    workload_service: WorkloadService = Depends(get_workload_service),
    risk_service: RiskService = Depends(get_risk_service),
    rag_engine: RAGDecisionEngine = Depends(get_rag_engine),
):
    """Submits a natural language query and returns an evidence-backed recommendation with scientific citations."""
    athlete = repo.get_athlete(request.athlete_id)
    if not athlete:
        raise HTTPException(status_code=404, detail=f"Athlete '{request.athlete_id}' not found.")

    history = repo.get_telemetry_history(request.athlete_id)
    try:
        workload = workload_service.compute_summary(history)
        risk = risk_service.evaluate_risk(history, target_sleep_hours=athlete.target_sleep_hours)
    except InsufficientDataError as e:
        raise HTTPException(status_code=400, detail=str(e.message)) from e

    return rag_engine.generate_decision(
        athlete_id=request.athlete_id,
        query_text=request.query_text,
        workload=workload,
        risk=risk,
    )
