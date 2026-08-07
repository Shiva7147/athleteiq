"""SQLAlchemy Implementation of AthleteRepository.

Provides thread-safe data access mapping between SQLAlchemy ORM tables and Pydantic DTOs.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.database.schema import AthleteORM, TelemetryORM
from src.models import AthleteCreate, AthleteRead, TelemetryCreate, TelemetryRead
from src.repositories.base import BaseAthleteRepository
from src.utils.exceptions import DatabaseError


class SQLAlchemyAthleteRepository(BaseAthleteRepository):
    """SQLAlchemy data access repository implementing `BaseAthleteRepository`."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_athlete(self, athlete: AthleteCreate) -> AthleteRead:
        """Stores a new athlete profile in the database.

        Args:
            athlete: `AthleteCreate` DTO.

        Returns:
            `AthleteRead` DTO.
        """
        try:
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
            self.session.add(db_obj)
            self.session.commit()
            self.session.refresh(db_obj)
            return AthleteRead.model_validate(db_obj)
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseError("Failed to create athlete in database.", details={"error": str(e)}) from e

    def get_athlete(self, athlete_id: str) -> Optional[AthleteRead]:
        """Retrieves an athlete profile by ID.

        Args:
            athlete_id: Athlete string ID.

        Returns:
            Optional `AthleteRead` DTO.
        """
        try:
            db_obj = self.session.query(AthleteORM).filter(AthleteORM.id == athlete_id).first()
            if not db_obj:
                return None
            return AthleteRead.model_validate(db_obj)
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to query athlete '{athlete_id}'.", details={"error": str(e)}) from e

    def list_athletes(self) -> List[AthleteRead]:
        """Lists all registered athlete profiles sorted by name.

        Returns:
            List of `AthleteRead` DTOs.
        """
        try:
            results = self.session.query(AthleteORM).order_by(AthleteORM.name.asc()).all()
            return [AthleteRead.model_validate(obj) for obj in results]
        except SQLAlchemyError as e:
            raise DatabaseError("Failed to list athletes.", details={"error": str(e)}) from e

    def log_telemetry(self, telemetry: TelemetryCreate) -> TelemetryRead:
        """Logs daily wearable telemetry and training session load.

        Args:
            telemetry: `TelemetryCreate` DTO.

        Returns:
            `TelemetryRead` DTO.
        """
        try:
            record_id = f"TEL-{uuid.uuid4().hex[:8].upper()}"
            session_load = telemetry.rpe_score * telemetry.session_duration_minutes

            # Remove existing record if same date exists
            self.session.query(TelemetryORM).filter(
                TelemetryORM.athlete_id == telemetry.athlete_id,
                TelemetryORM.recorded_date == telemetry.recorded_date,
            ).delete()

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
            self.session.add(db_obj)
            self.session.commit()
            self.session.refresh(db_obj)
            return TelemetryRead.model_validate(db_obj)
        except SQLAlchemyError as e:
            self.session.rollback()
            raise DatabaseError("Failed to log telemetry entry.", details={"error": str(e)}) from e

    def get_telemetry_history(self, athlete_id: str) -> List[TelemetryRead]:
        """Retrieves chronologically sorted telemetry history for an athlete.

        Args:
            athlete_id: Athlete string ID.

        Returns:
            List of `TelemetryRead` DTOs sorted by recorded_date ascending.
        """
        try:
            results = (
                self.session.query(TelemetryORM)
                .filter(TelemetryORM.athlete_id == athlete_id)
                .order_by(TelemetryORM.recorded_date.asc())
                .all()
            )
            return [TelemetryRead.model_validate(obj) for obj in results]
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to fetch telemetry history for '{athlete_id}'.", details={"error": str(e)}) from e
