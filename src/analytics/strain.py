"""Training Monotony and Strain Index Module (Foster's Method).

Pure deterministic Python functions for computing workload variability.
"""

import math
from typing import List


def calculate_monotony(daily_loads_7d: List[float]) -> float:
    """Calculates Training Monotony (Foster's Method: Mean / StdDev over 7 days).

    High monotony (> 2.0) combined with high load indicates elevated risk of overtraining.
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
    """Calculates Training Strain (Foster's Method: Total Weekly Load * Monotony)."""
    total_weekly_load = sum(daily_loads_7d)
    monotony = calculate_monotony(daily_loads_7d)
    return round(total_weekly_load * monotony, 2)
