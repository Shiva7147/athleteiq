"""Sports Science Workload Analytics Engine for AthleteIQ Pro.

Implements Acute-to-Chronic Workload Ratio (ACWR), Exponentially Weighted Moving
Average (EWMA), Training Monotony, and Training Strain Index algorithms.
"""

import math
from typing import List

from athleteiq.core.exceptions import InsufficientDataError
from athleteiq.core.models import DailyTelemetry, WorkloadMetrics
from athleteiq.data.validation import validate_daily_telemetry


def calculate_acute_workload(daily_loads: List[float]) -> float:
    """Calculates Acute Workload over a 7-day rolling window.

    Acute workload reflects short-term fatigue and recent training stress.

    Args:
        daily_loads: List of daily sRPE load values for the past 7 days.

    Returns:
        Mean daily load over the 7-day acute window.
    """
    if not daily_loads:
        return 0.0
    return sum(daily_loads[-7:]) / min(len(daily_loads), 7)


def calculate_chronic_workload(daily_loads: List[float]) -> float:
    """Calculates Chronic Workload over a 28-day rolling window.

    Chronic workload represents long-term fitness and aerobic capacity.

    Args:
        daily_loads: List of daily sRPE load values for the past 28 days.

    Returns:
        Mean daily load over the 28-day chronic window.
    """
    if not daily_loads:
        return 0.0
    return sum(daily_loads[-28:]) / min(len(daily_loads), 28)


def calculate_acwr_uncoupled(acute_workload: float, chronic_workload: float) -> float:
    """Calculates uncoupled Acute-to-Chronic Workload Ratio (ACWR).

    Args:
        acute_workload: 7-day acute workload value.
        chronic_workload: 28-day chronic workload value.

    Returns:
        ACWR ratio (Acute / Chronic). Returns 0.0 if chronic workload is zero.
    """
    if chronic_workload <= 0.0:
        return 0.0
    return round(acute_workload / chronic_workload, 3)


def calculate_ewma(daily_loads: List[float], time_decay_days: int) -> float:
    """Calculates Exponentially Weighted Moving Average (EWMA) for a load series.

    EWMA weights recent daily loads higher than older loads using a decay parameter:
    lambda = 2 / (time_decay_days + 1)

    Args:
        daily_loads: Sequential list of daily loads ordered from oldest to newest.
        time_decay_days: Window length N (e.g. 7 for acute, 28 for chronic).

    Returns:
        Final EWMA load value.
    """
    if not daily_loads:
        return 0.0

    decay_factor = 2.0 / (time_decay_days + 1.0)
    ewma_val = daily_loads[0]

    for current_load in daily_loads[1:]:
        ewma_val = (current_load * decay_factor) + (ewma_val * (1.0 - decay_factor))

    return ewma_val


def calculate_acwr_ewma(daily_loads: List[float]) -> float:
    """Calculates EWMA-based ACWR ratio.

    Compares 7-day EWMA acute load to 28-day EWMA chronic load.

    Args:
        daily_loads: Sequential list of daily loads (minimum 28 days recommended).

    Returns:
        EWMA ACWR ratio. Returns 0.0 if chronic EWMA is zero.
    """
    acute_ewma = calculate_ewma(daily_loads, time_decay_days=7)
    chronic_ewma = calculate_ewma(daily_loads, time_decay_days=28)

    if chronic_ewma <= 0.0:
        return 0.0

    return round(acute_ewma / chronic_ewma, 3)


def calculate_monotony(daily_loads_7d: List[float]) -> float:
    """Calculates Training Monotony (Foster's Method).

    Monotony = Mean(daily_loads) / Standard_Deviation(daily_loads)
    High monotony (> 2.0) combined with high load indicates elevated risk.

    Args:
        daily_loads_7d: List of daily loads over a 7-day window.

    Returns:
        Monotony score. Returns 0.0 if standard deviation is zero or insufficient data.
    """
    if len(daily_loads_7d) < 2:
        return 0.0

    mean_load = sum(daily_loads_7d) / len(daily_loads_7d)
    variance = sum((x - mean_load) ** 2 for x in daily_loads_7d) / len(daily_loads_7d)
    std_dev = math.sqrt(variance)

    if std_dev <= 1e-6:
        return 0.0

    return round(mean_load / std_dev, 3)


def calculate_strain(daily_loads_7d: List[float]) -> float:
    """Calculates Training Strain (Foster's Method).

    Strain = Total_Weekly_Load * Monotony

    Args:
        daily_loads_7d: List of daily loads over a 7-day window.

    Returns:
        Calculated strain index.
    """
    total_weekly_load = sum(daily_loads_7d)
    monotony = calculate_monotony(daily_loads_7d)
    return round(total_weekly_load * monotony, 2)


def compute_workload_summary(telemetry_history: List[DailyTelemetry]) -> WorkloadMetrics:
    """Computes comprehensive workload analytics DTO for an athlete.

    Args:
        telemetry_history: Historical list of `DailyTelemetry` records, ordered chronologically.

    Returns:
        A populated `WorkloadMetrics` DTO.

    Raises:
        InsufficientDataError: If history contains fewer than 28 days.
    """
    if len(telemetry_history) < 28:
        raise InsufficientDataError(
            f"Workload summary requires at least 28 days of history. Received {len(telemetry_history)} days.",
            details={"record_count": len(telemetry_history)},
        )

    # Sort telemetry by recorded date
    sorted_history = sorted(telemetry_history, key=lambda t: t.recorded_date)

    # Validate all telemetry entries defensively
    for record in sorted_history:
        validate_daily_telemetry(record)

    daily_loads = [t.session_load for t in sorted_history]
    recent_7d_loads = daily_loads[-7:]

    acute = calculate_acute_workload(daily_loads)
    chronic = calculate_chronic_workload(daily_loads)
    acwr_uncoupled = calculate_acwr_uncoupled(acute, chronic)
    acwr_ewma = calculate_acwr_ewma(daily_loads)
    monotony = calculate_monotony(recent_7d_loads)
    strain = calculate_strain(recent_7d_loads)

    latest_record = sorted_history[-1]

    return WorkloadMetrics(
        athlete_id=latest_record.athlete_id,
        as_of_date=latest_record.recorded_date,
        acute_workload_7d=round(acute, 2),
        chronic_workload_28d=round(chronic, 2),
        acwr_uncoupled=acwr_uncoupled,
        acwr_ewma=acwr_ewma,
        monotony=monotony,
        strain=strain,
    )
