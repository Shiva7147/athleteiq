"""Unit tests for Telemetry Data Validation engine."""

from datetime import date
import pytest

from athleteiq.core.exceptions import InvalidTelemetryError
from athleteiq.core.models import DailyTelemetry
from athleteiq.data.validation import validate_daily_telemetry


def test_validate_valid_telemetry() -> None:
    """Verifies that valid physiological telemetry passes without raising exceptions."""
    valid_record = DailyTelemetry(
        athlete_id="ATH-100",
        recorded_date=date(2026, 8, 1),
        hr_rest_bpm=52,
        hrv_rmssd_ms=72.5,
        sleep_hours=7.5,
        rpe_score=6.0,
        session_duration_minutes=60.0,
        total_distance_meters=8500.0,
        high_speed_running_meters=450.0,
    )
    # Should not raise any exception
    validate_daily_telemetry(valid_record)


def test_validate_invalid_rpe_raises_error() -> None:
    """Verifies that RPE > 10.0 raises InvalidTelemetryError."""
    invalid_record = DailyTelemetry(
        athlete_id="ATH-100",
        recorded_date=date(2026, 8, 1),
        hr_rest_bpm=52,
        hrv_rmssd_ms=72.5,
        sleep_hours=7.5,
        rpe_score=11.5,  # Out of range
        session_duration_minutes=60.0,
    )
    with pytest.raises(InvalidTelemetryError) as exc_info:
        validate_daily_telemetry(invalid_record)
    assert "RPE score" in str(exc_info.value)


def test_validate_invalid_rhr_raises_error() -> None:
    """Verifies that resting HR out of [30, 220] raises InvalidTelemetryError."""
    invalid_record = DailyTelemetry(
        athlete_id="ATH-100",
        recorded_date=date(2026, 8, 1),
        hr_rest_bpm=25,  # Below 30
        hrv_rmssd_ms=72.5,
        sleep_hours=7.5,
        rpe_score=5.0,
        session_duration_minutes=60.0,
    )
    with pytest.raises(InvalidTelemetryError) as exc_info:
        validate_daily_telemetry(invalid_record)
    assert "Resting HR" in str(exc_info.value)
