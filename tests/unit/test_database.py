"""Unit tests for Database ORM schemas and connection initialization."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import Base
from src.database.schema import AthleteORM, TelemetryORM


def test_schema_metadata_creation() -> None:
    """Verifies creating tables and indexes on an in-memory SQLite engine."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    athlete = AthleteORM(
        id="ATH-100",
        name="Test Player",
        sport="Football",
        position="Striker",
        age=22,
    )
    session.add(athlete)
    session.commit()

    retrieved = session.query(AthleteORM).filter_by(id="ATH-100").first()
    assert retrieved is not None
    assert retrieved.name == "Test Player"
    session.close()
