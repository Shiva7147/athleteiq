"""Telemetry Data Validation Engine for AthleteIQ Pro.

Enforces production data contracts and domain rules on raw wearable telemetry
before downstream processing by analytics or machine learning modules.
"""

from athleteiq.core.exceptions import InvalidTelemetryError
from athleteiq.core.models import DailyTelemetry


def validate_daily_telemetry(telemetry: DailyTelemetry) -> None:
    """Validates daily athlete telemetry against physiological boundaries.

    Args:
        telemetry: A `DailyTelemetry` object containing daily metrics.

    Raises:
        InvalidTelemetryError: If any metric falls outside accepted sports science ranges.
    """
    if not (1.0 <= telemetry.rpe_score <= 10.0):
        raise InvalidTelemetryError(
            f"RPE score {telemetry.rpe_score} is out of valid range [1.0, 10.0].",
            details={"rpe_score": telemetry.rpe_score, "athlete_id": telemetry.athlete_id},
        )

    if not (30 <= telemetry.hr_rest_bpm <= 220):
        raise InvalidTelemetryError(
            f"Resting HR {telemetry.hr_rest_bpm} bpm is physiologically improbable [30, 220].",
            details={"hr_rest_bpm": telemetry.hr_rest_bpm, "athlete_id": telemetry.athlete_id},
        )

    if not (5.0 <= telemetry.hrv_rmssd_ms <= 300.0):
        raise InvalidTelemetryError(
            f"HRV rMSSD {telemetry.hrv_rmssd_ms} ms is out of valid bounds [5.0, 300.0].",
            details={"hrv_rmssd_ms": telemetry.hrv_rmssd_ms, "athlete_id": telemetry.athlete_id},
        )

    if not (0.0 <= telemetry.sleep_hours <= 24.0):
        raise InvalidTelemetryError(
            f"Sleep hours {telemetry.sleep_hours} must be between 0.0 and 24.0.",
            details={"sleep_hours": telemetry.sleep_hours, "athlete_id": telemetry.athlete_id},
        )

    if telemetry.session_duration_minutes < 0.0:
        raise InvalidTelemetryError(
            f"Session duration {telemetry.session_duration_minutes} cannot be negative.",
            details={
                "session_duration_minutes": telemetry.session_duration_minutes,
                "athlete_id": telemetry.athlete_id,
            },
        )

    if telemetry.total_distance_meters < 0.0:
        raise InvalidTelemetryError(
            f"Total distance {telemetry.total_distance_meters} cannot be negative.",
            details={
                "total_distance_meters": telemetry.total_distance_meters,
                "athlete_id": telemetry.athlete_id,
            },
        )

    if telemetry.high_speed_running_meters < 0.0:
        raise InvalidTelemetryError(
            f"High speed running distance {telemetry.high_speed_running_meters} cannot be negative.",
            details={
                "high_speed_running_meters": telemetry.high_speed_running_meters,
                "athlete_id": telemetry.athlete_id,
            },
        )
