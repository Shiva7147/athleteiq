"""sRPE (Session Rate of Perceived Exertion) Calculation Module.

Pure deterministic Python functions for computing sRPE internal load.
Formula: sRPE Load = RPE (1-10) * Duration (minutes).
"""

from src.utils.exceptions import InvalidTelemetryError


def calculate_srpe_load(rpe: float, duration_minutes: float) -> float:
    """Calculates sRPE internal training session load.

    Args:
        rpe: Session Rate of Perceived Exertion (Borg scale 1.0 to 10.0).
        duration_minutes: Training session duration in minutes.

    Returns:
        Calculated session load in Arbitrary Units (AU).

    Raises:
        InvalidTelemetryError: If RPE is not in [1.0, 10.0] or duration < 0.
    """
    if not (1.0 <= rpe <= 10.0):
        raise InvalidTelemetryError(
            f"RPE score {rpe} must be between 1.0 and 10.0.",
            details={"rpe": rpe},
        )
    if duration_minutes < 0.0:
        raise InvalidTelemetryError(
            f"Session duration {duration_minutes} cannot be negative.",
            details={"duration_minutes": duration_minutes},
        )
    return round(rpe * duration_minutes, 2)
