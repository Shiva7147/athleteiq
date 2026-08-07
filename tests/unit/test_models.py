"""Unit tests for Pydantic domain schemas and DTOs."""

from datetime import date
import pytest
from pydantic import ValidationError

from src.models import AthleteCreate, TelemetryCreate, TelemetryRead, ACWRRiskZone, RiskTier


def test_athlete_create_schema() -> None:
    """Verifies AthleteCreate validation."""
    athlete = AthleteCreate(name="David Miller", sport="Soccer", position="Defender", age=24)
    assert athlete.name == "David Miller"
    assert athlete.target_sleep_hours == 8.0


def test_telemetry_create_invalid_rpe_raises_error() -> None:
    """Verifies that RPE > 10 raises Pydantic ValidationError."""
    with pytest.raises(ValidationError):
        TelemetryCreate(
            athlete_id="ATH-001",
            recorded_date=date(2026, 8, 1),
            hr_rest_bpm=52,
            hrv_rmssd_ms=65.0,
            sleep_hours=8.0,
            rpe_score=11.5,  # Invalid
            session_duration_minutes=60.0,
        )


def test_telemetry_read_session_load() -> None:
    """Verifies TelemetryRead DTO session load output."""
    t_read = TelemetryRead(
        id="TEL-001",
        athlete_id="ATH-001",
        recorded_date=date(2026, 8, 1),
        hr_rest_bpm=52,
        hrv_rmssd_ms=65.0,
        sleep_hours=8.0,
        rpe_score=7.0,
        session_duration_minutes=90.0,
        session_load=630.0,
    )
    assert t_read.session_load == 630.0
