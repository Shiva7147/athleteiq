"""SQLite Database Connection and Session Engine for AthleteIQ Pro."""

import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

# Local SQLite database file path
DB_PATH = os.getenv("ATHLETEIQ_DB_PATH", "athleteiq.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Connect args check_same_thread=False allows multi-threaded FastAPI access to SQLite
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass


def init_db() -> None:
    """Initializes and creates all database tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency generator providing transactional database session for FastAPI endpoints."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
