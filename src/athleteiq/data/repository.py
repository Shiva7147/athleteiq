"""Athlete Telemetry and Baseline Data Repository Layer.

Implements repository interfaces and thread-safe in-memory data access objects
for managing daily athlete telemetry histories and physiological baselines.
"""

from abc import ABC, abstractmethod
import math
from typing import Dict, List, Optional

from athleteiq.core.exceptions import BaselineValidationError, InsufficientDataError
from athleteiq.core.models import AthleteBaseline, DailyTelemetry
from athleteiq.data.validation import validate_daily_telemetry


class AthleteRepository(ABC):
    """Abstract Base Class defining repository data access contracts for AthleteIQ Pro."""

    @abstractmethod
    def add_telemetry(self, record: DailyTelemetry) -> None:
        """Stores a validated daily telemetry record."""
        pass

    @abstractmethod
    def get_telemetry_history(self, athlete_id: str) -> List[DailyTelemetry]:
        """Retrieves all telemetry records for an athlete, sorted chronologically."""
        pass

    @abstractmethod
    def set_baseline(self, baseline: AthleteBaseline) -> None:
        """Stores an athlete's statistical baseline."""
        pass

    @abstractmethod
    def get_baseline(self, athlete_id: str) -> Optional[AthleteBaseline]:
        """Retrieves an athlete's baseline statistics."""
        pass


class InMemoryAthleteRepository(AthleteRepository):
    """Thread-safe in-memory implementation of `AthleteRepository`."""

    def __init__(self) -> None:
        self._telemetry_store: Dict[str, List[DailyTelemetry]] = {}
        self._baseline_store: Dict[str, AthleteBaseline] = {}

    def add_telemetry(self, record: DailyTelemetry) -> None:
        """Validates and appends a telemetry record to the athlete's history.

        Args:
            record: A `DailyTelemetry` record.
        """
        validate_daily_telemetry(record)
        history = self._telemetry_store.setdefault(record.athlete_id, [])

        # Overwrite record if date already exists in history
        history[:] = [r for r in history if r.recorded_date != record.recorded_date]
        history.append(record)
        history.sort(key=lambda r: r.recorded_date)

    def get_telemetry_history(self, athlete_id: str) -> List[DailyTelemetry]:
        """Retrieves chronologically sorted telemetry records for an athlete.

        Args:
            athlete_id: Unique string identifier.

        Returns:
            List of `DailyTelemetry` records.
        """
        return list(self._telemetry_store.get(athlete_id, []))

    def set_baseline(self, baseline: AthleteBaseline) -> None:
        """Stores an athlete's statistical baseline metrics.

        Args:
            baseline: `AthleteBaseline` DTO.
        """
        if baseline.std_hr_rest_bpm < 0.0 or baseline.std_hrv_rmssd_ms < 0.0:
            raise BaselineValidationError(
                "Standard deviations in baseline metrics cannot be negative.",
                details={"athlete_id": baseline.athlete_id},
            )
        self._baseline_store[baseline.athlete_id] = baseline

    def get_baseline(self, athlete_id: str) -> Optional[AthleteBaseline]:
        """Retrieves stored baseline metrics for an athlete.

        Args:
            athlete_id: Unique string identifier.

        Returns:
            Optional `AthleteBaseline` object.
        """
        return self._baseline_store.get(athlete_id)

    def compute_baseline_from_history(
        self, athlete_id: str, target_sleep_hours: float = 8.0
    ) -> AthleteBaseline:
        """Calculates statistical mean and standard deviation from historical telemetry.

        Args:
            athlete_id: Unique athlete identifier.
            target_sleep_hours: Clinical sleep target in hours.

        Returns:
            A populated `AthleteBaseline` object.

        Raises:
            InsufficientDataError: If history contains fewer than 14 records.
        """
        history = self.get_telemetry_history(athlete_id)
        if len(history) < 14:
            raise InsufficientDataError(
                f"Computing a baseline requires at least 14 days of history. Received {len(history)}.",
                details={"record_count": len(history)},
            )

        rhrs = [t.hr_rest_bpm for t in history]
        hrvs = [t.hrv_rmssd_ms for t in history]

        n = len(history)
        mean_rhr = sum(rhrs) / n
        mean_hrv = sum(hrvs) / n

        var_rhr = sum((x - mean_rhr) ** 2 for x in rhrs) / n
        var_hrv = sum((x - mean_hrv) ** 2 for x in hrvs) / n

        std_rhr = math.sqrt(var_rhr)
        std_hrv = math.sqrt(var_hrv)

        baseline = AthleteBaseline(
            athlete_id=athlete_id,
            mean_hr_rest_bpm=round(mean_rhr, 2),
            std_hr_rest_bpm=round(std_rhr, 2),
            mean_hrv_rmssd_ms=round(mean_hrv, 2),
            std_hrv_rmssd_ms=round(std_hrv, 2),
            target_sleep_hours=target_sleep_hours,
        )
        self.set_baseline(baseline)
        return baseline
