"""Athlete Domain Models and Pydantic Schemas."""

from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class AthleteBase(BaseModel):
    """Base fields for athlete profiles."""

    name: str = Field(..., description="Full name of athlete")
    sport: str = Field(..., description="Sport discipline")
    position: str = Field(..., description="Specialty or position")
    age: int = Field(..., ge=12, le=60, description="Age in years")
    target_sleep_hours: float = Field(8.0, ge=4.0, le=14.0, description="Target sleep baseline in hours")


class AthleteCreate(AthleteBase):
    """Schema for registering a new athlete."""

    pass


class AthleteRead(AthleteBase):
    """Schema for reading an athlete profile."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Unique athlete ID")
    created_at: datetime = Field(..., description="Profile creation timestamp")


class AthleteBaseline(BaseModel):
    """Historical 30-day baseline statistics for an athlete."""

    athlete_id: str
    mean_hr_rest_bpm: float
    std_hr_rest_bpm: float
    mean_hrv_rmssd_ms: float
    std_hrv_rmssd_ms: float
    target_sleep_hours: float = 8.0
