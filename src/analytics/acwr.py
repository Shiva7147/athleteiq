"""Acute-to-Chronic Workload Ratio (ACWR) Calculation Engine.

Pure deterministic Python functions for 7-day Acute Load, 28-day Chronic Load,
Uncoupled ACWR, EWMA ACWR, and Risk Zone mapping.
"""

from typing import List
from src.models import ACWRRiskZone


def calculate_acute_load(daily_loads: List[float]) -> float:
    """Calculates Acute Workload over a 7-day rolling window."""
    if not daily_loads:
        return 0.0
    acute_slice = daily_loads[-7:]
    return sum(acute_slice) / len(acute_slice)


def calculate_chronic_load(daily_loads: List[float]) -> float:
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
    """Calculates Exponentially Weighted Moving Average (EWMA) load.

    Decay parameter lambda = 2 / (window_days + 1).
    """
    if not daily_loads:
        return 0.0

    decay_lambda = 2.0 / (window_days + 1.0)
    ewma = daily_loads[0]

    for load in daily_loads[1:]:
        ewma = (load * decay_lambda) + (ewma * (1.0 - decay_lambda))

    return ewma


def calculate_ewma_acwr(daily_loads: List[float]) -> float:
    """Calculates EWMA-based ACWR ratio."""
    acute_ewma = calculate_ewma_load(daily_loads, window_days=7)
    chronic_ewma = calculate_ewma_load(daily_loads, window_days=28)

    if chronic_ewma <= 0.0:
        return 0.0

    return round(acute_ewma / chronic_ewma, 3)


def classify_risk_zone(acwr: float) -> ACWRRiskZone:
    """Maps ACWR ratio to sports science risk zone."""
    if acwr > 1.5:
        return ACWRRiskZone.DANGER_ZONE
    if acwr > 1.3:
        return ACWRRiskZone.ELEVATED_RISK
    if acwr >= 0.8:
        return ACWRRiskZone.SWEET_SPOT
    return ACWRRiskZone.UNDER_TRAINING
