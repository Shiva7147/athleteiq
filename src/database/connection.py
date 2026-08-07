"""Database Connection Engine and Session Management.

Uses centralized Pydantic settings for configuration and provides thread-safe session factories.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from src.config import settings

# Create SQLAlchemy engine using centralized database URL
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
    echo=settings.debug,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base ORM declarative class."""

    pass


def init_db() -> None:
    """Creates database tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


def get_db_session() -> Generator[Session, None, None]:
    """Dependency generator supplying transactional database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
