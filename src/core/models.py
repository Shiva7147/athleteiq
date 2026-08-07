"""Core Domain Models for ACWR Calculation Engine.

Provides immutable Value Objects representing daily training workloads,
ACWR mathematical results, and sports science risk zones.
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum


class ACWRRiskZone(str, Enum):
    """Sports science risk zone categorization based on ACWR ratio."""

    UNDER_TRAINING = "UNDER_TRAINING"  # ACWR < 0.8: Fitness decay / under-preparation
    SWEET_SPOT = "SWEET_SPOT"          # 0.8 <= ACWR <= 1.3: Optimal workload / low injury risk
    ELEVATED_RISK = "ELEVATED_RISK"    # 1.3 < ACWR <= 1.5: Moderately elevated injury risk
    DANGER_ZONE = "DANGER_ZONE"        # ACWR > 1.5: High soft-tissue injury risk spike


@dataclass(frozen=True)
class WorkloadEntry:
    """Immutable record of an individual training session load.

    Attributes:
        entry_date: Date of training session.
        rpe: Session Rate of Perceived Exertion (Borg scale 1.0 to 10.0).
        duration_minutes: Session duration in minutes.
    """

    entry_date: date
    rpe: float
    duration_minutes: float

    @property
    def session_load(self) -> float:
        """Calculates sRPE Session Load (RPE * Duration)."""
        return self.rpe * self.duration_minutes


@dataclass(frozen=True)
class ACWRResult:
    """Comprehensive output DTO from an ACWR engine calculation.

    Attributes:
        athlete_id: Unique string identifier for the athlete.
        evaluation_date: Date corresponding to the calculation window end.
        acute_workload: 7-day mean session load.
        chronic_workload: 28-day mean session load.
        acwr_uncoupled: Simple ratio of acute to chronic workload.
        acwr_ewma: Exponentially Weighted Moving Average ACWR ratio.
        risk_zone: Categorical sports science risk zone.
    """

    athlete_id: str
    evaluation_date: date
    acute_workload: float
    chronic_workload: float
    acwr_uncoupled: float
    acwr_ewma: float
    risk_zone: ACWRRiskZone
