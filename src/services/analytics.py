"""Deterministic Workload Analytics Calculation Engine.

Calculates sRPE, 7-day Acute Workload, 28-day Chronic Workload, Uncoupled ACWR,
EWMA ACWR, Foster's Monotony, and Foster's Strain.
"""

import math
from typing import List

from src.core.exceptions import InsufficientDataError
from src.domain.schemas import TelemetryRead, WorkloadSummaryRead


def calculate_acute_workload(daily_loads: List[float]) -> float:
    """Calculates Acute Workload over a 7-day rolling window."""
    if not daily_loads:
        return 0.0
    acute_slice = daily_loads[-7:]
    return sum(acute_slice) / len(acute_slice)


def calculate_chronic_workload(daily_loads: List[float]) -> float:
    """Calculates Chronic Workload over a 28-day rolling window."""
    if not daily_loads:
        return 0.0
    chronic_slice = daily_loads[-28:]
    return sum(chronic_slice) / len(chronic_slice)


def calculate_uncoupled_acwr(acute: float, chronic: float) -> float:
    """Calculates Uncoupled ACWR (Acute / Chronic)."""
    if chronic <= 0.0:
        return 0.0
    return round(acute / chronic, 3)


def calculate_ewma_load(daily_loads: List[float], window_days: int) -> float:
    """Calculates EWMA load using decay lambda = 2 / (window_days + 1)."""
    if not daily_loads:
        return 0.0

    decay_lambda = 2.0 / (window_days + 1.0)
    ewma = daily_loads[0]

    for load in daily_loads[1:]:
        ewma = (load * decay_lambda) + (ewma * (1.0 - decay_lambda))

    return ewma


def calculate_ewma_acwr(daily_loads: List[float]) -> float:
    """Calculates EWMA ACWR ratio."""
    acute_ewma = calculate_ewma_load(daily_loads, window_days=7)
    chronic_ewma = calculate_ewma_load(daily_loads, window_days=28)

    if chronic_ewma <= 0.0:
        return 0.0

    return round(acute_ewma / chronic_ewma, 3)


def calculate_monotony(daily_loads_7d: List[float]) -> float:
    """Calculates Training Monotony (Foster's Method: Mean / StdDev)."""
    if len(daily_loads_7d) < 2:
        return 0.0

    mean_load = sum(daily_loads_7d) / len(daily_loads_7d)
    variance = sum((x - mean_load) ** 2 for x in daily_loads_7d) / len(daily_loads_7d)
    std_dev = math.sqrt(variance)

    if std_dev <= 1e-6:
        return 0.0

    return round(mean_load / std_dev, 3)


def calculate_strain(daily_loads_7d: List[float]) -> float:
    """Calculates Training Strain (Foster's Method: Total Weekly Load * Monotony)."""
    total_weekly_load = sum(daily_loads_7d)
    monotony = calculate_monotony(daily_loads_7d)
    return round(total_weekly_load * monotony, 2)


def classify_risk_zone(acwr: float) -> str:
    """Maps ACWR ratio to sports science risk zone."""
    if acwr > 1.5:
        return "DANGER_ZONE"
    if acwr > 1.3:
        return "ELEVATED_RISK"
    if acwr >= 0.8:
        return "SWEET_SPOT"
    return "UNDER_TRAINING"


def compute_workload_summary(telemetry_history: List[TelemetryRead]) -> WorkloadSummaryRead:
    """Computes full deterministic workload summary from telemetry records.

    Args:
        telemetry_history: Historical list of `TelemetryRead` objects.

    Returns:
        `WorkloadSummaryRead` DTO.

    Raises:
        InsufficientDataError: If history contains fewer than 28 days.
    """
    if len(telemetry_history) < 28:
        raise InsufficientDataError(
            f"Workload analytics require at least 28 days of telemetry history. Received {len(telemetry_history)}.",
            details={"received_records": len(telemetry_history)},
        )

    sorted_history = sorted(telemetry_history, key=lambda t: t.recorded_date)
    daily_loads = [t.session_load for t in sorted_history]
    recent_7d = daily_loads[-7:]

    acute = calculate_acute_workload(daily_loads)
    chronic = calculate_chronic_workload(daily_loads)
    acwr_uncoupled = calculate_uncoupled_acwr(acute, chronic)
    acwr_ewma = calculate_ewma_acwr(daily_loads)
    monotony = calculate_monotony(recent_7d)
    strain = calculate_strain(recent_7d)
    risk_zone = classify_risk_zone(acwr_ewma)

    latest = sorted_history[-1]

    return WorkloadSummaryRead(
        athlete_id=latest.athlete_id,
        evaluation_date=latest.recorded_date,
        acute_workload_7d=round(acute, 2),
        chronic_workload_28d=round(chronic, 2),
        acwr_uncoupled=acwr_uncoupled,
        acwr_ewma=acwr_ewma,
        monotony=monotony,
        strain=strain,
        risk_zone=risk_zone,
    )
