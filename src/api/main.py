"""FastAPI Production Backend REST API for AthleteIQ Pro.

Provides enterprise REST endpoints for athlete management, session logging,
deterministic workload analytics, injury risk assessments, and RAG decision support.
"""

from typing import List
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.db.database import get_db, init_db
from src.db.repository import create_athlete, get_athlete, get_telemetry_history, list_athletes, log_telemetry
from src.domain.schemas import (
    AthleteCreate,
    AthleteRead,
    DecisionQueryRequest,
    RiskAssessmentRead,
    StructuredDecisionResponse,
    TelemetryCreate,
    TelemetryRead,
    WorkloadSummaryRead,
)
from src.services.analytics import compute_workload_summary
from src.services.risk_engine import evaluate_risk
from src.ai.rag_engine import generate_structured_decision

# Initialize Database tables
init_db()

app = FastAPI(
    title="AthleteIQ Pro API",
    description="Production-quality AI Decision Support Platform for Sports Science.",
    version="1.0.0",
)

# Enable CORS for Streamlit / Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "AthleteIQ Pro API", "version": "1.0.0"}


@app.post("/api/v1/athletes", response_model=AthleteRead, status_code=status.HTTP_201_CREATED, tags=["Athletes"])
def api_create_athlete(athlete: AthleteCreate, db: Session = Depends(get_db)):
    """Creates a new athlete profile."""
    return create_athlete(db, athlete)


@app.get("/api/v1/athletes", response_model=List[AthleteRead], tags=["Athletes"])
def api_list_athletes(db: Session = Depends(get_db)):
    """Lists all registered athlete profiles."""
    return list_athletes(db)


@app.get("/api/v1/athletes/{athlete_id}", response_model=AthleteRead, tags=["Athletes"])
def api_get_athlete(athlete_id: str, db: Session = Depends(get_db)):
    """Retrieves an athlete profile by ID."""
    athlete = get_athlete(db, athlete_id)
    if not athlete:
        raise HTTPException(status_code=404, detail=f"Athlete with ID '{athlete_id}' not found.")
    return athlete


@app.post("/api/v1/telemetry", response_model=TelemetryRead, status_code=status.HTTP_201_CREATED, tags=["Telemetry"])
def api_log_telemetry(telemetry: TelemetryCreate, db: Session = Depends(get_db)):
    """Logs a daily wearable telemetry entry."""
    athlete = get_athlete(db, telemetry.athlete_id)
    if not athlete:
        raise HTTPException(status_code=404, detail=f"Athlete with ID '{telemetry.athlete_id}' not found.")
    return log_telemetry(db, telemetry)


@app.get("/api/v1/telemetry/{athlete_id}", response_model=List[TelemetryRead], tags=["Telemetry"])
def api_get_telemetry(athlete_id: str, db: Session = Depends(get_db)):
    """Retrieves chronologically sorted telemetry history for an athlete."""
    return get_telemetry_history(db, athlete_id)


@app.get("/api/v1/analytics/{athlete_id}", response_model=WorkloadSummaryRead, tags=["Analytics"])
def api_get_analytics(athlete_id: str, db: Session = Depends(get_db)):
    """Computes deterministic workload summary (ACWR, EWMA, Monotony, Strain)."""
    history = get_telemetry_history(db, athlete_id)
    if len(history) < 28:
        raise HTTPException(
            status_code=400,
            detail=f"Deterministic workload analytics require at least 28 days of history. Found {len(history)} records.",
        )
    return compute_workload_summary(history)


@app.get("/api/v1/risk/{athlete_id}", response_model=RiskAssessmentRead, tags=["Risk Engine"])
def api_get_risk(athlete_id: str, db: Session = Depends(get_db)):
    """Runs deterministic injury risk assessment."""
    athlete = get_athlete(db, athlete_id)
    if not athlete:
        raise HTTPException(status_code=404, detail=f"Athlete '{athlete_id}' not found.")
    history = get_telemetry_history(db, athlete_id)
    if len(history) < 28:
        raise HTTPException(
            status_code=400,
            detail=f"Risk assessment requires at least 28 days of history. Found {len(history)} records.",
        )
    return evaluate_risk(history, target_sleep_hours=athlete.target_sleep_hours)


@app.post("/api/v1/decisions/query", response_model=StructuredDecisionResponse, tags=["AI Decision Support"])
def api_query_decision(request: DecisionQueryRequest, db: Session = Depends(get_db)):
    """Submits a natural language query and returns an evidence-backed structured recommendation with citations."""
    athlete = get_athlete(db, request.athlete_id)
    if not athlete:
        raise HTTPException(status_code=404, detail=f"Athlete '{request.athlete_id}' not found.")

    history = get_telemetry_history(db, request.athlete_id)
    if len(history) < 28:
        raise HTTPException(
            status_code=400,
            detail=f"AI Decision Support requires at least 28 days of history. Found {len(history)} records.",
        )

    workload = compute_workload_summary(history)
    risk = evaluate_risk(history, target_sleep_hours=athlete.target_sleep_hours)

    return generate_structured_decision(
        athlete_id=request.athlete_id,
        query_text=request.query_text,
        workload=workload,
        risk=risk,
    )
