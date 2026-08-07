"""Pydantic V2 Domain Schemas and Data Transfer Objects (DTOs) for AthleteIQ Pro.

Defines enterprise API contracts for athletes, telemetry session logs,
deterministic workload analytics, injury risk assessments, scientific citations,
and structured decision-support responses.
"""

from datetime import date, datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class RiskTier(str, Enum):
    """Categorical injury risk levels based on ACWR and physiological stress markers."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AthleteBase(BaseModel):
    """Base fields for an athlete profile."""

    name: str = Field(..., description="Full name of the athlete")
    sport: str = Field(..., description="Sport discipline (e.g. Football, Basketball, Track)")
    position: str = Field(..., description="Position or specialization (e.g. Midfielder, Point Guard, Sprinter)")
    age: int = Field(..., ge=12, le=60, description="Age in years")
    target_sleep_hours: float = Field(8.0, ge=4.0, le=14.0, description="Clinical target sleep duration in hours")


class AthleteCreate(AthleteBase):
    """Schema for creating a new athlete profile."""

    pass


class AthleteRead(AthleteBase):
    """Schema for reading an athlete profile."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Unique UUID or string identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of profile creation")


class TelemetryCreate(BaseModel):
    """Schema for logging daily wearable telemetry and training session load."""

    athlete_id: str = Field(..., description="Target athlete identifier")
    recorded_date: date = Field(..., description="Date of the telemetry record")
    hr_rest_bpm: int = Field(..., ge=30, le=220, description="Resting heart rate in bpm")
    hrv_rmssd_ms: float = Field(..., ge=5.0, le=300.0, description="Heart Rate Variability rMSSD in milliseconds")
    sleep_hours: float = Field(..., ge=0.0, le=24.0, description="Hours of sleep duration")
    rpe_score: float = Field(..., ge=1.0, le=10.0, description="Rate of Perceived Exertion (Borg scale 1-10)")
    session_duration_minutes: float = Field(..., ge=0.0, le=600.0, description="Session duration in minutes")
    total_distance_meters: float = Field(0.0, ge=0.0, description="Total GPS distance covered in meters")
    high_speed_running_meters: float = Field(0.0, ge=0.0, description="High-speed running distance (>19.8 km/h) in meters")
    injuries_reported: Optional[str] = Field(None, description="Subjective injury or soreness reports")


class TelemetryRead(TelemetryCreate):
    """Schema for reading a recorded telemetry entry."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Unique record identifier")
    session_load: float = Field(..., description="Calculated sRPE session load (RPE * Duration)")


class WorkloadSummaryRead(BaseModel):
    """Deterministic workload analytics summary."""

    model_config = ConfigDict(from_attributes=True)

    athlete_id: str
    evaluation_date: date
    acute_workload_7d: float
    chronic_workload_28d: float
    acwr_uncoupled: float
    acwr_ewma: float
    monotony: float
    strain: float
    risk_zone: str


class RiskAssessmentRead(BaseModel):
    """Deterministic injury risk assessment output."""

    model_config = ConfigDict(from_attributes=True)

    athlete_id: str
    assessment_date: date
    risk_tier: RiskTier
    risk_score: float
    contributing_factors: List[str] = Field(default_factory=list)
    recommended_action: str


class Citation(BaseModel):
    """Peer-reviewed sports science literature citation."""

    title: str
    authors: str
    journal: str
    year: int
    key_finding: str
    doi_or_link: str


class DecisionQueryRequest(BaseModel):
    """Coach's natural language question request."""

    athlete_id: str
    query_text: str


class StructuredDecisionResponse(BaseModel):
    """Evidence-backed structured recommendation response for coaches."""

    athlete_id: str
    query_text: str
    deterministic_acwr: float
    deterministic_risk_tier: RiskTier
    summary_recommendation: str
    action_points: List[str]
    citations: List[Citation]
