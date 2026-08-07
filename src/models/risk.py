"""Risk Domain Models and Enums."""

from datetime import date
from enum import Enum
from typing import List
from pydantic import BaseModel, Field, ConfigDict


class RiskTier(str, Enum):
    """Categorical injury risk levels based on physiological and workload stress markers."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class InjuryRiskAssessmentRead(BaseModel):
    """Deterministic soft-tissue injury risk assessment DTO."""

    model_config = ConfigDict(from_attributes=True)

    athlete_id: str
    assessment_date: date
    risk_tier: RiskTier
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Calibrated risk probability score [0.0, 1.0]")
    contributing_factors: List[str] = Field(default_factory=list, description="Top physiological stress drivers")
    recommended_action: str = Field(..., description="Actionable clinical recommendation for coaches")
