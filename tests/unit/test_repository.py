"""Unit tests for SQLAlchemyAthleteRepository."""

from datetime import date
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.models import AthleteCreate, TelemetryCreate
from src.repositories.athlete_repo import SQLAlchemyAthleteRepository


@pytest.fixture
def repo_session():
    """Provides a fresh isolated in-memory database session for each test."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    repo = SQLAlchemyAthleteRepository(session)
    yield repo
    session.close()


def test_repository_create_and_get_athlete(repo_session: SQLAlchemyAthleteRepository) -> None:
    """Tests creating and fetching an athlete using repository pattern."""
    created = repo_session.create_athlete(
        AthleteCreate(name="Lucas Silva", sport="Soccer", position="Midfielder", age=25)
    )
    assert created.id.startswith("ATH-")
    assert created.name == "Lucas Silva"

    retrieved = repo_session.get_athlete(created.id)
    assert retrieved is not None
    assert retrieved.name == "Lucas Silva"


def test_repository_log_telemetry_and_history(repo_session: SQLAlchemyAthleteRepository) -> None:
    """Tests logging daily telemetry and querying chronological history."""
    athlete = repo_session.create_athlete(
        AthleteCreate(name="Lucas Silva", sport="Soccer", position="Midfielder", age=25)
    )
    t = repo_session.log_telemetry(
        TelemetryCreate(
            athlete_id=athlete.id,
            recorded_date=date(2026, 8, 1),
            hr_rest_bpm=52,
            hrv_rmssd_ms=68.0,
            sleep_hours=8.0,
            rpe_score=6.5,
            session_duration_minutes=90.0,
        )
    )
    assert t.session_load == 585.0  # 6.5 * 90.0

    history = repo_session.get_telemetry_history(athlete.id)
    assert len(history) == 1
    assert history[0].session_load == 585.0
