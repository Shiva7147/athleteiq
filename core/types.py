"""
core/types.py
Pydantic v2 schemas for domain data types and sports science metrics.
"""
from datetime import date
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class MatchLeverage(str, Enum):
    """
    Tier-based match leverage classification.
    """
    TIER_1_MUST_WIN = "tier_1_must_win"       # Finals, title-deciders, derbies, knockout
    TIER_2_STANDARD = "tier_2_standard"       # Regular league matches
    TIER_3_ROTATION = "tier_3_rotation"       # Dead-rubbers, low-stakes cup fixtures


class SessionType(str, Enum):
    TRAINING = "training"
    MATCH = "match"
    RECOVERY = "recovery"


class DailySessionMetrics(BaseModel):
    """
    GPS and sensor session record for a single athlete on a specific date.
    """
    model_config = ConfigDict(strict=True)

    player_id: str
    session_date: date
    session_type: SessionType
    minutes_played: int = Field(ge=0, le=130)
    total_distance_m: float = Field(ge=0.0)
    high_speed_running_m: float = Field(ge=0.0, description="Distance covered at > 19.8 km/h")
    sprint_distance_m: float = Field(ge=0.0, description="Distance covered at > 25.2 km/h")
    accelerations_count: int = Field(ge=0, description="Accelerations > 3.0 m/s²")
    rpe: float = Field(ge=0.0, le=10.0, description="Rate of Perceived Exertion (Borg CR10)")


class DailyWellness(BaseModel):
    """
    Morning athlete self-reported wellness and autonomic recovery.
    """
    model_config = ConfigDict(strict=True)

    player_id: str
    record_date: date
    sleep_hours: float = Field(ge=0.0, le=16.0)
    soreness_score: int = Field(ge=1, le=5, description="1=Very sore, 5=Fresh")
    fatigue_score: int = Field(ge=1, le=5, description="1=Exhausted, 5=Fully energetic")
    stress_score: int = Field(ge=1, le=5, description="1=High stress, 5=Calm")
    mood_score: int = Field(ge=1, le=5, description="1=Poor, 5=Great")
    hrv_rmssd: Optional[float] = Field(default=None, ge=10.0, le=250.0, description="Resting RMSSD (ms)")
    resting_hr_bpm: Optional[int] = Field(default=None, ge=30, le=120)


class MatchScheduleEntry(BaseModel):
    """
    Context for an upcoming fixture.
    """
    model_config = ConfigDict(strict=True)

    fixture_id: str
    match_date: date
    opponent: str
    leverage_tier: MatchLeverage
    importance_multiplier: float = Field(
        default=1.0, ge=0.5, le=2.5,
        description="Weight of the match in tactical vs risk decision-making"
    )
