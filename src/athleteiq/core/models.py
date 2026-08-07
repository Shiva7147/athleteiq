"""Core Domain Models for AthleteIQ Pro.

Implements immutable Value Objects and Data Transfer Objects (DTOs)
representing raw telemetry, workload metrics, and injury risk assessments.
"""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import List, Optional


class RiskTier(str, Enum):
    """Categorical injury risk levels based on ACWR and physiological stress markers."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class DailyTelemetry:
    """Raw daily wearable and subjective telemetry for an individual athlete.

    Attributes:
        athlete_id: Unique string identifier for the athlete.
        recorded_date: Calendar date of the telemetry record.
        hr_rest_bpm: Resting Heart Rate in beats per minute.
        hrv_rmssd_ms: Heart Rate Variability (rMSSD) in milliseconds.
        sleep_hours: Total duration of sleep in hours.
        rpe_score: Rate of Perceived Exertion (Borg scale 1-10).
        session_duration_minutes: Training session duration in minutes.
        total_distance_meters: Total GPS distance covered in meters.
        high_speed_running_meters: High-speed running distance (>19.8 km/h) in meters.
    """

    athlete_id: str
    recorded_date: date
    hr_rest_bpm: int
    hrv_rmssd_ms: float
    sleep_hours: float
    rpe_score: float
    session_duration_minutes: float
    total_distance_meters: float = 0.0
    high_speed_running_meters: float = 0.0

    @property
    def session_load(self) -> float:
        """Calculates sRPE (Session Rate of Perceived Exertion) Load.

        Formula: sRPE Load = RPE Score (1-10) * Duration (minutes).
        Standard metric in sports science for internal training load.
        """
        return self.rpe_score * self.session_duration_minutes


@dataclass(frozen=True)
class WorkloadMetrics:
    """Calculated workload analytics for an athlete over rolling windows.

    Attributes:
        athlete_id: Unique identifier for the athlete.
        as_of_date: Date corresponding to the end of the evaluation window.
        acute_workload_7d: 7-day rolling average session load (Short-term fatigue).
        chronic_workload_28d: 28-day rolling average session load (Long-term fitness).
        acwr_uncoupled: Standard ratio of 7-day load to 28-day load (Acute / Chronic).
        acwr_ewma: Exponentially Weighted Moving Average ACWR (gives higher weight to recent load).
        monotony: Training monotony index (Mean daily load / Standard deviation of daily load over 7 days).
        strain: Training strain index (Total weekly load * Monotony).
    """

    athlete_id: str
    as_of_date: date
    acute_workload_7d: float
    chronic_workload_28d: float
    acwr_uncoupled: float
    acwr_ewma: float
    monotony: float
    strain: float


@dataclass(frozen=True)
class InjuryRiskAssessment:
    """Comprehensive AI decision-support output for injury risk prediction.

    Attributes:
        athlete_id: Unique athlete identifier.
        assessment_date: Evaluation date.
        risk_tier: Categorical risk level (LOW, MODERATE, HIGH, CRITICAL).
        risk_score: Normalized probabilistic risk score between 0.0 and 1.0.
        contributing_factors: Key physiological or workload drivers increasing risk.
        recommended_action: Actionable clinical recommendation for coaches/trainers.
    """

    athlete_id: str
    assessment_date: date
    risk_tier: RiskTier
    risk_score: float
    contributing_factors: List[str] = field(default_factory=list)
    recommended_action: str = ""
