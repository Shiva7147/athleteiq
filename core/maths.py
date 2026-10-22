"""
core/maths.py
Deterministic, unit-tested sports-science mathematical primitives.

Formulas:
1. Acute Exponentially Weighted Moving Average (EWMA):
   EWMA_t = Load_t * lambda + (1 - lambda) * EWMA_{t-1}
   where lambda = 2 / (N + 1)
   Default: N = 7 days for Acute, N = 28 days for Chronic.

2. Acute:Chronic Workload Ratio (Coupled & Uncoupled EWMA):
   ACWR = Acute_EWMA / Chronic_EWMA

3. Foster's Training Monotony:
   Monotony = Mean(7d Daily Loads) / StdDev(7d Daily Loads)
   High monotony (> 2.0) with high strain reflects monotonous training without recovery days.

4. Foster's Training Strain:
   Strain = Sum(7d Daily Loads) * Monotony

5. Match Leverage Index (MLI) & Risk Tolerance Adjustment:
   Effective Actionable Risk = Physical Calibrated Risk / MLI
   In a Final (MLI = 2.0), acceptable risk threshold is higher, triaging staff attention.
"""
from typing import Sequence
import numpy as np


def compute_ewma_series(values: Sequence[float], time_decay_days: int) -> np.ndarray:
    """
    Computes Exponentially Weighted Moving Average with strict point-in-time sequential flow.
    
    Args:
        values: Chronological sequence of daily loads (Day 0, Day 1, ... Day T).
        time_decay_days: Time constant N (e.g. 7 for acute, 28 for chronic).
        
    Returns:
        np.ndarray of same length containing EWMA at each day t.
    """
    if len(values) == 0:
        return np.array([], dtype=np.float64)
    
    if time_decay_days < 1:
        raise ValueError("time_decay_days must be >= 1")

    decay_lambda = 2.0 / (time_decay_days + 1.0)
    arr = np.asarray(values, dtype=np.float64)
    ewma = np.zeros(len(arr), dtype=np.float64)
    ewma[0] = arr[0]
    
    for i in range(1, len(arr)):
        ewma[i] = arr[i] * decay_lambda + (1.0 - decay_lambda) * ewma[i - 1]
        
    return ewma


def compute_acwr(acute_load: float, chronic_load: float) -> float:
    """
    Computes Acute:Chronic Workload Ratio.
    Handles zero chronic load edge cases gracefully by capping safely.
    """
    if chronic_load <= 1e-4:
        return 1.0 if acute_load <= 1e-4 else 2.5
    return float(acute_load / chronic_load)


def compute_training_monotony(daily_loads_7d: Sequence[float]) -> float:
    """
    Foster's Training Monotony over a 7-day period.
    Monotony = Mean / Standard Deviation.
    """
    arr = np.asarray(daily_loads_7d, dtype=np.float64)
    if len(arr) != 7:
        raise ValueError(f"Monotony requires exactly 7 days of load, received {len(arr)}")
    
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))
    
    if std_val < 1e-4:
        # If daily load is identical across all 7 days with non-zero mean, monotony is infinite/capped
        return 10.0 if mean_val > 0.0 else 0.0
    
    return float(mean_val / std_val)


def compute_training_strain(daily_loads_7d: Sequence[float]) -> float:
    """
    Foster's Training Strain = Weekly Workload * Monotony.
    """
    arr = np.asarray(daily_loads_7d, dtype=np.float64)
    monotony = compute_training_monotony(arr)
    weekly_sum = float(np.sum(arr))
    return float(weekly_sum * monotony)


def calculate_match_leverage_weight(leverage_tier: str) -> float:
    """
    Maps match importance tier into Match Leverage Index (MLI).
    Tier 1 (Final/Must Win): 2.0
    Tier 2 (Standard League): 1.0
    Tier 3 (Dead Rubber/Rotation): 0.6
    """
    tier_map = {
        "tier_1_must_win": 2.0,
        "tier_2_standard": 1.0,
        "tier_3_rotation": 0.6,
    }
    return tier_map.get(leverage_tier.lower(), 1.0)


def evaluate_decision_action(
    calibrated_physical_risk: float,
    match_leverage_multiplier: float,
    alert_budget_cutoff: float = 0.15
) -> dict:
    """
    Combines physical probability with tactical leverage.
    
    Returns structured decision recommendation:
    - CLEAR_TO_PLAY
    - PLAY_WITH_MINUTE_CAP (e.g. 60 min substitution limit)
    - ROTATE_AND_PRESCRIBE_MICRODOSE (rest match, but sprint microdose to avoid under-prep)
    - CLINICAL_RED_FLAG (immediate benching / scan required)
    """
    if calibrated_physical_risk >= 0.35:
        return {
            "decision": "CLINICAL_REST_MANDATORY",
            "reason": "Extreme tissue overload risk regardless of match tier.",
            "max_minutes": 0,
            "prescribe_microdose": False,
        }
    
    # If high leverage (Final), threshold for complete benching is higher, but minute-capping is applied
    if match_leverage_multiplier >= 1.8:
        if calibrated_physical_risk > 0.18:
            return {
                "decision": "PLAY_WITH_MINUTE_CAP",
                "reason": "High-leverage fixture. Athlete cleared for start; cap at 60 mins to prevent late eccentric fatigue breakdown.",
                "max_minutes": 60,
                "prescribe_microdose": False,
            }
        return {
            "decision": "CLEAR_TO_PLAY_FULL",
            "reason": "Acceptable risk profile for championship-tier fixture.",
            "max_minutes": 90,
            "prescribe_microdose": False,
        }
    
    # For routine league or rotation matches, protect the asset
    if calibrated_physical_risk >= alert_budget_cutoff:
        return {
            "decision": "ROTATE_AND_PRESCRIBE_MICRODOSE",
            "reason": "Fixture stakes do not justify elevated strain. Bench for match, but perform controlled high-speed running micro-dose.",
            "max_minutes": 0,
            "prescribe_microdose": True,
        }
        
    return {
        "decision": "CLEAR_TO_PLAY_FULL",
        "reason": "Within baseline safety limits.",
        "max_minutes": 90,
        "prescribe_microdose": False,
    }
