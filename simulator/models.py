"""
simulator/models.py
Data structures and rules for the multi-world synthetic sports league generator.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional


class SimulationWorld(str, Enum):
    """
    Five distinct causal worlds designed to prevent model circularity.
    """
    WORLD_A = "world_a_cumulative_fatigue"  # Fatigue builds steadily over multi-week microcycles
    WORLD_B = "world_b_under_preparation"    # Risk triggers when match sprint exposure exceeds training baseline
    WORLD_C = "world_c_interactions"        # Synergistic interaction: Poor sleep + load spike triggers injury
    WORLD_D = "world_d_delayed_effects"     # Non-linear latency: Load spikes trigger breakdown 10-21 days later
    WORLD_E = "world_e_mostly_noise"        # Stochastic, weak signal (mirrors real-world high unpredictability)


@dataclass
class PlayerProfile:
    """
    Simulated athlete bio-mechanical and physiological baseline profile.
    """
    player_id: str
    name: str
    position: str  # 'CB', 'FB', 'CM', 'W', 'ST'
    age: int
    base_fitness: float          # 50.0 to 90.0
    tissue_resilience: float     # 0.7 (fragile) to 1.3 (robust)
    max_sprint_velocity_kmh: float # 30.0 to 36.0 km/h
    prior_injury_count: int = 0
    days_lost_historical: int = 0
