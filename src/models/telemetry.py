"""Telemetry Domain Models and Validation Schemas."""

from datetime import date
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class TelemetryBase(BaseModel):
    """Base fields for daily wearable and session telemetry."""

    athlete_id: str = Field(..., description="Target athlete identifier")
    recorded_date: date = Field(..., description="Telemetry date")
    hr_rest_bpm: int = Field(..., ge=30, le=220, description="Resting HR in bpm")
    hrv_rmssd_ms: float = Field(..., ge=5.0, le=300.0, description="HRV rMSSD in ms")
    sleep_hours: float = Field(..., ge=0.0, le=24.0, description="Hours of sleep duration")
    rpe_score: float = Field(..., ge=1.0, le=10.0, description="Session RPE score (1-10)")
    session_duration_minutes: float = Field(..., ge=0.0, le=600.0, description="Session duration in minutes")
    total_distance_meters: float = Field(0.0, ge=0.0, description="GPS total distance covered in meters")
    high_speed_running_meters: float = Field(0.0, ge=0.0, description="High speed running distance (>19.8 km/h)")
    injuries_reported: Optional[str] = Field(None, description="Subjective injury / soreness notes")


class TelemetryCreate(TelemetryBase):
    """Schema for creating daily telemetry record."""

    pass


class TelemetryRead(TelemetryBase):
    """Schema for reading daily telemetry record."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Unique record identifier")
    session_load: float = Field(..., description="Calculated sRPE session load (RPE * Duration)")
