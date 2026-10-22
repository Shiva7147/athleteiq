"""
simulator/engine.py
Generates simulated seasons across distinct causal worlds.
Ensures the ML model does not merely memorize the simulator's internal math.
"""
from datetime import date, timedelta
from typing import List, Dict, Any
import numpy as np
from simulator.models import PlayerProfile, SimulationWorld
from core.types import MatchLeverage


class LeagueSimulator:
    """
    Simulates a full season (or microcycle) of training, matches, wellness, and injuries.
    """
    def __init__(self, world: SimulationWorld, random_seed: int = 42):
        self.world = world
        self.rng = np.random.default_rng(random_seed)

    def generate_squad(self, squad_size: int = 26) -> List[PlayerProfile]:
        """Generates a realistic 26-man squad with positional profiles."""
        positions = ["GK"] * 3 + ["CB"] * 5 + ["FB"] * 4 + ["CM"] * 6 + ["W"] * 4 + ["ST"] * 4
        squad: List[PlayerProfile] = []
        
        for i, pos in enumerate(positions[:squad_size]):
            player = PlayerProfile(
                player_id=f"PLY_{i+1:03d}",
                name=f"Athlete_{i+1}",
                position=pos,
                age=int(self.rng.integers(18, 35)),
                base_fitness=float(self.rng.normal(70.0, 8.0)),
                tissue_resilience=float(self.rng.uniform(0.75, 1.25)),
                max_sprint_velocity_kmh=float(self.rng.normal(32.5, 2.0)),
                prior_injury_count=int(self.rng.poisson(1.2)),
            )
            squad.append(player)
        return squad

    def simulate_day_microcycle(
        self,
        current_date: date,
        squad: List[PlayerProfile],
        is_match_day: bool,
        match_leverage: MatchLeverage = MatchLeverage.TIER_2_STANDARD
    ) -> List[Dict[str, Any]]:
        """
        Simulates load, wellness, and injury status for all squad players on a given day.
        """
        day_records: List[Dict[str, Any]] = []

        for p in squad:
            # 1. Simulate Session Metrics
            if p.position == "GK":
                distance = float(self.rng.normal(4500.0, 500.0))
                hsr = float(self.rng.normal(50.0, 20.0))
                sprints = float(self.rng.normal(10.0, 5.0))
                minutes = 90 if is_match_day else 75
            else:
                if is_match_day:
                    minutes = int(self.rng.choice([90, 65, 30, 0], p=[0.45, 0.20, 0.15, 0.20]))
                    distance = (minutes / 90.0) * float(self.rng.normal(10500.0, 800.0))
                    hsr = (minutes / 90.0) * float(self.rng.normal(750.0, 120.0))
                    sprints = (minutes / 90.0) * float(self.rng.normal(250.0, 50.0))
                else:
                    minutes = 75
                    distance = float(self.rng.normal(5500.0, 600.0))
                    hsr = float(self.rng.normal(350.0, 80.0))
                    sprints = float(self.rng.normal(80.0, 30.0))

            rpe = float(np.clip(self.rng.normal(7.0 if is_match_day else 5.0, 1.0), 1.0, 10.0))

            # 2. Simulate Wellness
            sleep_hours = float(np.clip(self.rng.normal(7.8, 0.9), 4.5, 10.5))
            fatigue_score = int(np.clip(self.rng.integers(1, 6), 1, 5))
            soreness_score = int(np.clip(self.rng.integers(1, 6), 1, 5))
            hrv_rmssd = float(np.clip(self.rng.normal(65.0, 15.0), 20.0, 140.0))

            # 3. Compute True Injury Probability based on World Causal Rule
            true_risk = self._compute_world_injury_risk(
                p=p,
                minutes=minutes,
                hsr=hsr,
                sprints=sprints,
                sleep_hours=sleep_hours,
                rpe=rpe,
                is_match_day=is_match_day
            )

            # Determine stochastic occurrence
            injured_today = bool(self.rng.uniform(0.0, 1.0) < true_risk)

            day_records.append({
                "date": current_date,
                "player_id": p.player_id,
                "position": p.position,
                "minutes": minutes,
                "total_distance_m": round(distance, 1),
                "high_speed_running_m": round(hsr, 1),
                "sprint_distance_m": round(sprints, 1),
                "rpe": round(rpe, 1),
                "sleep_hours": round(sleep_hours, 1),
                "fatigue_score": fatigue_score,
                "soreness_score": soreness_score,
                "hrv_rmssd": round(hrv_rmssd, 1),
                "true_injury_prob": round(true_risk, 4),
                "injury_event": injured_today,
            })

        return day_records

    def _compute_world_injury_risk(
        self,
        p: PlayerProfile,
        minutes: int,
        hsr: float,
        sprints: float,
        sleep_hours: float,
        rpe: float,
        is_match_day: bool
    ) -> float:
        """Applies the specific causal driver of the active simulation world."""
        base_rate = 0.008  # Baseline ~0.8% daily probability

        if self.world == SimulationWorld.WORLD_A:
            # Fatigue builds up: RPE * minutes
            load_factor = (rpe * minutes) / 600.0
            return float(np.clip(base_rate * (1.0 + load_factor) / p.tissue_resilience, 0.001, 0.35))

        elif self.world == SimulationWorld.WORLD_B:
            # Under-preparation: High match minutes with low sprint baseline
            if is_match_day and minutes > 60 and sprints < 120.0:
                return float(np.clip(base_rate * 4.2 / p.tissue_resilience, 0.001, 0.40))
            return base_rate

        elif self.world == SimulationWorld.WORLD_C:
            # Interaction: Poor sleep (<6h) AND high match load
            if sleep_hours < 6.0 and minutes >= 75:
                return float(np.clip(base_rate * 5.5 / p.tissue_resilience, 0.001, 0.50))
            return base_rate

        elif self.world == SimulationWorld.WORLD_E:
            # Mostly noise: Weak signal, high stochastic randomness
            noise_factor = float(self.rng.uniform(0.5, 1.8))
            return float(np.clip(base_rate * noise_factor, 0.001, 0.08))

        return base_rate
