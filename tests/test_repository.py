"""Unit tests for Athlete Data Repository Layer."""

from datetime import date, timedelta
import pytest

from athleteiq.core.exceptions import BaselineValidationError, InsufficientDataError
from athleteiq.core.models import AthleteBaseline, DailyTelemetry
from athleteiq.data.repository import InMemoryAthleteRepository


def test_repository_add_and_get_telemetry() -> None:
    """Verifies storing and retrieving chronologically sorted telemetry records."""
    repo = InMemoryAthleteRepository()

    t1 = DailyTelemetry(
        athlete_id="ATH-001",
        recorded_date=date(2026, 8, 2),
        hr_rest_bpm=55,
        hrv_rmssd_ms=60.0,
        sleep_hours=8.0,
        rpe_score=6.0,
        session_duration_minutes=60.0,
    )
    t0 = DailyTelemetry(
        athlete_id="ATH-001",
        recorded_date=date(2026, 8, 1),
        hr_rest_bpm=54,
        hrv_rmssd_ms=62.0,
        sleep_hours=7.5,
        rpe_score=5.0,
        session_duration_minutes=60.0,
    )

    repo.add_telemetry(t1)
    repo.add_telemetry(t0)

    history = repo.get_telemetry_history("ATH-001")
    assert len(history) == 2
    # Should be sorted chronologically (Aug 1 before Aug 2)
    assert history[0].recorded_date == date(2026, 8, 1)
    assert history[1].recorded_date == date(2026, 8, 2)


def test_repository_set_and_get_baseline() -> None:
    """Verifies storing and retrieving athlete baseline statistics."""
    repo = InMemoryAthleteRepository()
    baseline = AthleteBaseline(
        athlete_id="ATH-001",
        mean_hr_rest_bpm=52.0,
        std_hr_rest_bpm=3.5,
        mean_hrv_rmssd_ms=68.0,
        std_hrv_rmssd_ms=7.2,
    )
    repo.set_baseline(baseline)
    retrieved = repo.get_baseline("ATH-001")
    assert retrieved == baseline


def test_repository_set_invalid_baseline_raises_error() -> None:
    """Verifies that negative standard deviation in baseline raises BaselineValidationError."""
    repo = InMemoryAthleteRepository()
    invalid_baseline = AthleteBaseline(
        athlete_id="ATH-001",
        mean_hr_rest_bpm=52.0,
        std_hr_rest_bpm=-2.0,  # Invalid
        mean_hrv_rmssd_ms=68.0,
        std_hrv_rmssd_ms=7.2,
    )
    with pytest.raises(BaselineValidationError):
        repo.set_baseline(invalid_baseline)


def test_compute_baseline_from_history() -> None:
    """Verifies automatic baseline calculation from 14+ telemetry history records."""
    repo = InMemoryAthleteRepository()
    for i in range(15):
        t = DailyTelemetry(
            athlete_id="ATH-002",
            recorded_date=date(2026, 8, 1) + timedelta(days=i),
            hr_rest_bpm=50 + (i % 3),
            hrv_rmssd_ms=70.0 + (i % 5),
            sleep_hours=8.0,
            rpe_score=5.0,
            session_duration_minutes=60.0,
        )
        repo.add_telemetry(t)

    baseline = repo.compute_baseline_from_history("ATH-002")
    assert baseline.athlete_id == "ATH-002"
    assert baseline.mean_hr_rest_bpm > 45.0
    assert baseline.std_hrv_rmssd_ms > 0.0
