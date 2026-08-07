"""Deterministic Acute-to-Chronic Workload Ratio (ACWR) Calculation Engine.

Provides high-performance, pure Python routines for calculating sRPE,
7-day acute workload, 28-day chronic workload, uncoupled ACWR, EWMA ACWR,
and sports science injury risk zone classification.
"""

from typing import List

from src.core.exceptions import InsufficientDataError, InvalidWorkloadValueError
from src.core.models import ACWRResult, ACWRRiskZone, WorkloadEntry


def validate_workload_entry(entry: WorkloadEntry) -> None:
    """Validates an individual workload entry against physiological boundaries.

    Args:
        entry: `WorkloadEntry` object.

    Raises:
        InvalidWorkloadValueError: If RPE is not in [1.0, 10.0] or duration < 0.
    """
    if not (1.0 <= entry.rpe <= 10.0):
        raise InvalidWorkloadValueError(
            f"RPE value {entry.rpe} must be between 1.0 and 10.0.",
            details={"rpe": entry.rpe, "date": entry.entry_date},
        )
    if entry.duration_minutes < 0.0:
        raise InvalidWorkloadValueError(
            f"Duration {entry.duration_minutes} cannot be negative.",
            details={"duration_minutes": entry.duration_minutes, "date": entry.entry_date},
        )


def calculate_acute_load(daily_loads: List[float]) -> float:
    """Calculates Acute Workload over a 7-day rolling window.

    Args:
        daily_loads: List of daily sRPE load values.

    Returns:
        Mean daily load over the acute window (7 days).
    """
    if not daily_loads:
        return 0.0
    acute_slice = daily_loads[-7:]
    return sum(acute_slice) / len(acute_slice)


def calculate_chronic_load(daily_loads: List[float]) -> float:
    """Calculates Chronic Workload over a 28-day rolling window.

    Args:
        daily_loads: List of daily sRPE load values.

    Returns:
        Mean daily load over the chronic window (28 days).
    """
    if not daily_loads:
        return 0.0
    chronic_slice = daily_loads[-28:]
    return sum(chronic_slice) / len(chronic_slice)


def calculate_uncoupled_acwr(acute: float, chronic: float) -> float:
    """Calculates Uncoupled ACWR (Acute / Chronic).

    Args:
        acute: 7-day acute workload.
        chronic: 28-day chronic workload.

    Returns:
        ACWR ratio rounded to 3 decimal places. Returns 0.0 if chronic <= 0.
    """
    if chronic <= 0.0:
        return 0.0
    return round(acute / chronic, 3)


def calculate_ewma_load(daily_loads: List[float], window_days: int) -> float:
    """Calculates Exponentially Weighted Moving Average (EWMA) load.

    Decay parameter lambda = 2 / (window_days + 1).

    Args:
        daily_loads: Sequential daily load values ordered from oldest to newest.
        window_days: Time decay window length N (e.g., 7 for acute, 28 for chronic).

    Returns:
        Final EWMA load value.
    """
    if not daily_loads:
        return 0.0

    decay_lambda = 2.0 / (window_days + 1.0)
    ewma = daily_loads[0]

    for load in daily_loads[1:]:
        ewma = (load * decay_lambda) + (ewma * (1.0 - decay_lambda))

    return ewma


def calculate_ewma_acwr(daily_loads: List[float]) -> float:
    """Calculates EWMA-based ACWR ratio.

    Compares 7-day EWMA acute load to 28-day EWMA chronic load.

    Args:
        daily_loads: Sequential list of daily loads.

    Returns:
        EWMA ACWR ratio rounded to 3 decimal places.
    """
    acute_ewma = calculate_ewma_load(daily_loads, window_days=7)
    chronic_ewma = calculate_ewma_load(daily_loads, window_days=28)

    if chronic_ewma <= 0.0:
        return 0.0

    return round(acute_ewma / chronic_ewma, 3)


def classify_risk_zone(acwr: float) -> ACWRRiskZone:
    """Categorizes ACWR ratio into a sports science risk zone.

    Args:
        acwr: Calculated ACWR ratio.

    Returns:
        `ACWRRiskZone` Enum value.
    """
    if acwr > 1.5:
        return ACWRRiskZone.DANGER_ZONE
    if acwr > 1.3:
        return ACWRRiskZone.ELEVATED_RISK
    if acwr >= 0.8:
        return ACWRRiskZone.SWEET_SPOT
    return ACWRRiskZone.UNDER_TRAINING


def calculate_acwr(athlete_id: str, entries: List[WorkloadEntry]) -> ACWRResult:
    """Computes comprehensive ACWR metrics and risk zone for an athlete.

    Args:
        athlete_id: Unique string identifier for the athlete.
        entries: Historical list of `WorkloadEntry` records (at least 28 days required).

    Returns:
        Populated `ACWRResult` DTO.

    Raises:
        InsufficientDataError: If history contains fewer than 28 days.
        InvalidWorkloadValueError: If any entry violates physiological bounds.
    """
    if len(entries) < 28:
        raise InsufficientDataError(
            f"ACWR calculation requires at least 28 days of entries. Received {len(entries)}.",
            details={"received_entries": len(entries), "athlete_id": athlete_id},
        )

    # Sort entries chronologically
    sorted_entries = sorted(entries, key=lambda e: e.entry_date)

    # Validate all entries
    for entry in sorted_entries:
        validate_workload_entry(entry)

    daily_loads = [e.session_load for e in sorted_entries]

    acute = calculate_acute_load(daily_loads)
    chronic = calculate_chronic_load(daily_loads)
    acwr_uncoupled = calculate_uncoupled_acwr(acute, chronic)
    acwr_ewma = calculate_ewma_acwr(daily_loads)
    risk_zone = classify_risk_zone(acwr_ewma)

    latest_entry = sorted_entries[-1]

    return ACWRResult(
        athlete_id=athlete_id,
        evaluation_date=latest_entry.entry_date,
        acute_workload=round(acute, 2),
        chronic_workload=round(chronic, 2),
        acwr_uncoupled=acwr_uncoupled,
        acwr_ewma=acwr_ewma,
        risk_zone=risk_zone,
    )
