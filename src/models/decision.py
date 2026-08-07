"""AI Decision Support and Citation Schemas."""

from typing import List
from pydantic import BaseModel, Field
from src.models.risk import RiskTier


class Citation(BaseModel):
    """Peer-reviewed sports science literature citation."""

    title: str = Field(..., description="Paper title")
    authors: str = Field(..., description="Authors list")
    journal: str = Field(..., description="Journal name")
    year: int = Field(..., description="Publication year")
    key_finding: str = Field(..., description="Key sports science finding")
    doi_or_link: str = Field(..., description="DOI link")


class DecisionQueryRequest(BaseModel):
    """Coach's natural language question request."""

    athlete_id: str = Field(..., description="Target athlete ID")
    query_text: str = Field(..., description="Question prompt")


class StructuredDecisionResponse(BaseModel):
    """Evidence-backed structured recommendation response."""

    athlete_id: str
    query_text: str
    deterministic_acwr: float
    deterministic_risk_tier: RiskTier
    summary_recommendation: str
    action_points: List[str]
    citations: List[Citation]
