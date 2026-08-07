"""Workload Domain Models and Enums."""

from datetime import date
from enum import Enum
from pydantic import BaseModel, ConfigDict


class ACWRRiskZone(str, Enum):
    """Categorical risk zones based on ACWR ratio thresholds."""

    UNDER_TRAINING = "UNDER_TRAINING"  # ACWR < 0.8: Fitness decay
    SWEET_SPOT = "SWEET_SPOT"          # 0.8 <= ACWR <= 1.3: Optimal workload / low injury risk
    ELEVATED_RISK = "ELEVATED_RISK"    # 1.3 < ACWR <= 1.5: Moderately elevated risk
    DANGER_ZONE = "DANGER_ZONE"        # ACWR > 1.5: High soft-tissue injury risk spike


class WorkloadSummaryRead(BaseModel):
    """Deterministic workload analytics summary DTO."""

    model_config = ConfigDict(from_attributes=True)

    athlete_id: str
    evaluation_date: date
    acute_workload_7d: float
    chronic_workload_28d: float
    acwr_uncoupled: float
    acwr_ewma: float
    monotony: float
    strain: float
    risk_zone: ACWRRiskZone
