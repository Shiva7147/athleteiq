"""Unit tests for SQLite Database Repository CRUD operations."""

from datetime import date
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.database import Base
from src.db.repository import create_athlete, get_athlete, get_telemetry_history, list_athletes, log_telemetry
from src.domain.schemas import AthleteCreate, TelemetryCreate

# In-memory SQLite for fast testing
TEST_ENGINE = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)


@pytest.fixture(autouse=True)
def setup_test_db():
    """Sets up an in-memory SQLite database for each test."""
    Base.metadata.create_all(bind=TEST_ENGINE)
    yield
    Base.metadata.drop_all(bind=TEST_ENGINE)


def test_create_and_get_athlete() -> None:
    """Verifies creating and retrieving an athlete profile."""
    db = TestSessionLocal()
    try:
        dto = AthleteCreate(
            name="Jordan Reed",
            sport="Basketball",
            position="Guard",
            age=22,
            target_sleep_hours=8.5,
        )
        created = create_athlete(db, dto)
        assert created.name == "Jordan Reed"
        assert created.id.startswith("ATH-")

        retrieved = get_athlete(db, created.id)
        assert retrieved is not None
        assert retrieved.name == "Jordan Reed"
    finally:
        db.close()


def test_log_and_get_telemetry() -> None:
    """Verifies logging telemetry and calculating session load."""
    db = TestSessionLocal()
    try:
        athlete = create_athlete(
            db,
            AthleteCreate(name="Sam Miller", sport="Track", position="Sprinter", age=21),
        )
        t_dto = TelemetryCreate(
            athlete_id=athlete.id,
            recorded_date=date(2026, 8, 1),
            hr_rest_bpm=50,
            hrv_rmssd_ms=75.0,
            sleep_hours=8.0,
            rpe_score=7.0,
            session_duration_minutes=60.0,
        )
        logged = log_telemetry(db, t_dto)
        assert logged.session_load == 420.0  # 7.0 * 60.0

        history = get_telemetry_history(db, athlete.id)
        assert len(history) == 1
        assert history[0].session_load == 420.0
    finally:
        db.close()
