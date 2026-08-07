"""SQLite Data Repository Layer for AthleteIQ Pro.

Implements database CRUD access methods converting between SQLAlchemy ORM objects
and Pydantic domain schemas.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.orm import Session

from src.db.models import AthleteORM, TelemetryORM
from src.domain.schemas import AthleteCreate, AthleteRead, TelemetryCreate, TelemetryRead


def create_athlete(db: Session, athlete: AthleteCreate) -> AthleteRead:
    """Creates a new athlete profile record in SQLite.

    Args:
        db: SQLAlchemy DB Session.
        athlete: `AthleteCreate` DTO.

    Returns:
        `AthleteRead` DTO.
    """
    athlete_id = f"ATH-{uuid.uuid4().hex[:8].upper()}"
    db_obj = AthleteORM(
        id=athlete_id,
        name=athlete.name,
        sport=athlete.sport,
        position=athlete.position,
        age=athlete.age,
        target_sleep_hours=athlete.target_sleep_hours,
        created_at=datetime.now(timezone.utc),
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return AthleteRead.model_validate(db_obj)


def get_athlete(db: Session, athlete_id: str) -> Optional[AthleteRead]:
    """Retrieves an athlete profile by ID.

    Args:
        db: DB Session.
        athlete_id: Athlete string ID.

    Returns:
        Optional `AthleteRead` DTO.
    """
    db_obj = db.query(AthleteORM).filter(AthleteORM.id == athlete_id).first()
    if not db_obj:
        return None
    return AthleteRead.model_validate(db_obj)


def list_athletes(db: Session) -> List[AthleteRead]:
    """Lists all registered athlete profiles.

    Args:
        db: DB Session.

    Returns:
        List of `AthleteRead` DTOs.
    """
    results = db.query(AthleteORM).order_by(AthleteORM.name).all()
    return [AthleteRead.model_validate(obj) for obj in results]


def log_telemetry(db: Session, telemetry: TelemetryCreate) -> TelemetryRead:
    """Logs daily wearable telemetry and training session load.

    Args:
        db: DB Session.
        telemetry: `TelemetryCreate` DTO.

    Returns:
        `TelemetryRead` DTO.
    """
    record_id = f"TEL-{uuid.uuid4().hex[:8].upper()}"
    session_load = telemetry.rpe_score * telemetry.session_duration_minutes

    db_obj = TelemetryORM(
        id=record_id,
        athlete_id=telemetry.athlete_id,
        recorded_date=telemetry.recorded_date,
        hr_rest_bpm=telemetry.hr_rest_bpm,
        hrv_rmssd_ms=telemetry.hrv_rmssd_ms,
        sleep_hours=telemetry.sleep_hours,
        rpe_score=telemetry.rpe_score,
        session_duration_minutes=telemetry.session_duration_minutes,
        total_distance_meters=telemetry.total_distance_meters,
        high_speed_running_meters=telemetry.high_speed_running_meters,
        injuries_reported=telemetry.injuries_reported,
        session_load=session_load,
    )
    # Remove existing record if same athlete and date already present
    db.query(TelemetryORM).filter(
        TelemetryORM.athlete_id == telemetry.athlete_id,
        TelemetryORM.recorded_date == telemetry.recorded_date,
    ).delete()

    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return TelemetryRead.model_validate(db_obj)


def get_telemetry_history(db: Session, athlete_id: str) -> List[TelemetryRead]:
    """Retrieves chronologically sorted telemetry history for an athlete.

    Args:
        db: DB Session.
        athlete_id: Athlete string ID.

    Returns:
        List of `TelemetryRead` DTOs sorted by recorded_date ascending.
    """
    results = (
        db.query(TelemetryORM)
        .filter(TelemetryORM.athlete_id == athlete_id)
        .order_by(TelemetryORM.recorded_date.asc())
        .all()
    )
    return [TelemetryRead.model_validate(obj) for obj in results]
