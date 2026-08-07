"""SQLAlchemy Database Schema Definitions.

Implements normalized ORM models with composite index idx_telemetry_athlete_date
for optimal O(log N) historical window query performance.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship

from src.database.connection import Base


class AthleteORM(Base):
    """SQLAlchemy ORM table mapping for athlete profiles."""

    __tablename__ = "athletes"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    sport = Column(String, nullable=False)
    position = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    target_sleep_hours = Column(Float, nullable=False, default=8.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    telemetry_records = relationship("TelemetryORM", back_populates="athlete", cascade="all, delete-orphan")


class TelemetryORM(Base):
    """SQLAlchemy ORM table mapping for daily wearable telemetry logs.

    Includes composite index `idx_telemetry_athlete_date` to accelerate 28-day window queries.
    """

    __tablename__ = "telemetry_logs"

    id = Column(String, primary_key=True, index=True)
    athlete_id = Column(String, ForeignKey("athletes.id"), nullable=False, index=True)
    recorded_date = Column(Date, nullable=False, index=True)
    hr_rest_bpm = Column(Integer, nullable=False)
    hrv_rmssd_ms = Column(Float, nullable=False)
    sleep_hours = Column(Float, nullable=False)
    rpe_score = Column(Float, nullable=False)
    session_duration_minutes = Column(Float, nullable=False)
    total_distance_meters = Column(Float, default=0.0, nullable=False)
    high_speed_running_meters = Column(Float, default=0.0, nullable=False)
    injuries_reported = Column(Text, nullable=True)
    session_load = Column(Float, nullable=False)

    athlete = relationship("AthleteORM", back_populates="telemetry_records")

    __table_args__ = (
        Index("idx_telemetry_athlete_date", "athlete_id", "recorded_date"),
    )


class RiskLogORM(Base):
    """SQLAlchemy ORM table mapping for historical risk assessment logs."""

    __tablename__ = "risk_logs"

    id = Column(String, primary_key=True, index=True)
    athlete_id = Column(String, ForeignKey("athletes.id"), nullable=False, index=True)
    assessment_date = Column(Date, nullable=False, index=True)
    risk_tier = Column(String, nullable=False)
    risk_score = Column(Float, nullable=False)
    contributing_factors_json = Column(Text, nullable=False)
    recommended_action = Column(Text, nullable=False)
