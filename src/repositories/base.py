"""Abstract Repository Interface Contracts for Data Persistence.

Enforces SOLID Dependency Inversion Principle isolating database operations from domain services.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from src.models import AthleteCreate, AthleteRead, TelemetryCreate, TelemetryRead


class BaseAthleteRepository(ABC):
    """Abstract Base Class defining repository data access contracts."""

    @abstractmethod
    def create_athlete(self, athlete: AthleteCreate) -> AthleteRead:
        """Stores a new athlete profile."""
        pass

    @abstractmethod
    def get_athlete(self, athlete_id: str) -> Optional[AthleteRead]:
        """Retrieves an athlete profile by ID."""
        pass

    @abstractmethod
    def list_athletes(self) -> List[AthleteRead]:
        """Lists all registered athlete profiles."""
        pass

    @abstractmethod
    def log_telemetry(self, telemetry: TelemetryCreate) -> TelemetryRead:
        """Logs daily wearable telemetry record."""
        pass

    @abstractmethod
    def get_telemetry_history(self, athlete_id: str) -> List[TelemetryRead]:
        """Retrieves chronologically sorted telemetry records for an athlete."""
        pass
