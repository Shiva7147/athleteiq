"""FastAPI Dependency Injection Handlers.

Provides dependencies for database sessions, repository access objects, domain services,
and AI decision engines.
"""

from typing import Generator
from fastapi import Depends
from sqlalchemy.orm import Session

from src.ai.decision_engine import RAGDecisionEngine
from src.database.connection import get_db_session
from src.repositories.athlete_repo import SQLAlchemyAthleteRepository
from src.repositories.base import BaseAthleteRepository
from src.services.risk_service import RiskService
from src.services.workload_service import WorkloadService

# Global service singletons
_workload_service = WorkloadService()
_risk_service = RiskService(workload_service=_workload_service)
_rag_engine = RAGDecisionEngine()


def get_repository(db: Session = Depends(get_db_session)) -> BaseAthleteRepository:
    """Dependency injection handler providing athlete repository instance."""
    return SQLAlchemyAthleteRepository(db)


def get_workload_service() -> WorkloadService:
    """Dependency injection handler providing WorkloadService."""
    return _workload_service


def get_risk_service() -> RiskService:
    """Dependency injection handler providing RiskService."""
    return _risk_service


def get_rag_engine() -> RAGDecisionEngine:
    """Dependency injection handler providing RAGDecisionEngine."""
    return _rag_engine
