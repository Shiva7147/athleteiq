"""Unit tests for domain exception hierarchy."""

from src.utils.exceptions import AthleteIQError, InvalidTelemetryError, InsufficientDataError


def test_athlete_iq_error_formatting() -> None:
    """Verifies string formatting for AthleteIQError with details."""
    err = AthleteIQError("Validation failed", details={"rpe": 12.0})
    assert "Validation failed" in str(err)
    assert "rpe" in str(err)


def test_subclass_exceptions() -> None:
    """Verifies exception inheritance hierarchy."""
    err = InvalidTelemetryError("RPE invalid")
    assert isinstance(err, AthleteIQError)
    assert isinstance(err, Exception)
