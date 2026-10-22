"""
simulator/trajectory.py
Forward-rolling Banister/Busso impulse-response trajectory sandbox.
Simulates athlete physiological response (fitness, fatigue, readiness, and indicative risk)
under distinct hypothetical coaching decisions across multi-match horizons.

Scientific Disclaimer:
- Banister systems model macro performance readiness and fatigue accumulation, not clinical pathology.
- The 7-day injury risk curve represents a statistical decision-support heuristic,
  not an individual medical diagnosis or validated clinical prescription.
- Conformal bounds reflect empirical calibration spreads under exchangeability assumptions.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
from core.types import MatchLeverage
from ml.models import compute_conformal_bounds


@dataclass
class ScheduledFixture:
    """
    Fixture context on a specific future day offset.
    """
    day_offset: int           # 0 = Today, 1 = Tomorrow, etc.
    opponent: str
    leverage_tier: MatchLeverage
    is_match: bool = True


@dataclass
class ProposedAction:
    """
    What the coach proposes for the player on day_offset.
    """
    day_offset: int
    planned_minutes: int                    # 0 to 90+
    session_intensity: float                # 0.0 (rest/recovery) to 1.0 (match intensity)
    target_high_speed_running_m: float     # Target HSR meters


@dataclass
class DayTrajectorySnapshot:
    """
    Projected physiological state for a specific day.
    """
    day_offset: int
    projected_fatigue: float
    projected_fitness: float
    readiness_score: float                  # 0 to 100
    calibrated_injury_risk: float           # 0.0 to 1.0
    risk_ci_lower_95: float
    risk_ci_upper_95: float
    decision_recommendation: str
    rationale: str


class TrajectorySimulationEngine:
    """
    Simulates multi-day forward trajectories of athlete readiness and injury risk.
    Uses continuous-time impulse-response differential equations:
    Fatigue(t) = Fatigue(t-1) * exp(-1 / tau_fatigue) + k_fatigue * Impulse(t)
    Fitness(t) = Fitness(t-1) * exp(-1 / tau_fitness) + k_fitness * Impulse(t)
    Readiness(t) = Fitness(t) - Fatigue(t) + Baseline
    """
    def __init__(
        self,
        tau_fatigue: float = 7.0,     # Fatigue decays in ~7 days
        tau_fitness: float = 28.0,    # Fitness decays in ~28 days
        k_fatigue: float = 1.8,       # Fatigue impulse gain (builds faster than fitness)
        k_fitness: float = 1.0,       # Fitness impulse gain
    ):
        self.tau_fatigue = tau_fatigue
        self.tau_fitness = tau_fitness
        self.k_fatigue = k_fatigue
        self.k_fitness = k_fitness

    def simulate_trajectory(
        self,
        baseline_fitness: float,
        baseline_fatigue: float,
        schedule: List[ScheduledFixture],
        coaching_plan: List[ProposedAction],
        tissue_vulnerability_index: float = 1.0,  # History-adjusted multiplier
    ) -> List[DayTrajectorySnapshot]:
        """
        Executes a deterministic forward rollout across the planning horizon.
        """
        plan_map = {p.day_offset: p for p in coaching_plan}
        sched_map = {s.day_offset: s for s in schedule}
        horizon_days = max(max(plan_map.keys(), default=0), max(sched_map.keys(), default=0)) + 1

        snapshots: List[DayTrajectorySnapshot] = []
        curr_fitness = baseline_fitness
        curr_fatigue = baseline_fatigue

        for day in range(horizon_days):
            plan = plan_map.get(day, ProposedAction(day, 0, 0.2, 0.0))
            fixture = sched_map.get(day, None)

            # 1. State at start of day (Morning Evaluation)
            readiness = float(np.clip(curr_fitness - curr_fatigue + 50.0, 0.0, 100.0))

            # Calibrated probability model: non-linear logistic hazard
            risk_logit = -3.2 + (0.05 * curr_fatigue) - (0.02 * curr_fitness)
            if fixture and fixture.is_match and plan.planned_minutes > 45 and plan.target_high_speed_running_m < 50.0:
                risk_logit += 0.40

            prob_raw = 1.0 / (1.0 + np.exp(-risk_logit))
            calibrated_risk = float(np.clip(prob_raw * tissue_vulnerability_index, 0.005, 0.65))

            # Inductive conformal prediction intervals (nominal 95% coverage)
            ci_lower, ci_upper = compute_conformal_bounds(calibrated_risk, alpha=0.05)

            # Contextual Decision Generation
            decision, rationale = self._evaluate_tactical_decision(
                fixture=fixture,
                plan=plan,
                calibrated_risk=calibrated_risk
            )

            snapshots.append(DayTrajectorySnapshot(
                day_offset=day,
                projected_fatigue=round(curr_fatigue, 2),
                projected_fitness=round(curr_fitness, 2),
                readiness_score=round(readiness, 1),
                calibrated_injury_risk=round(calibrated_risk, 4),
                risk_ci_lower_95=round(ci_lower, 4),
                risk_ci_upper_95=round(ci_upper, 4),
                decision_recommendation=decision,
                rationale=rationale,
            ))

            # 2. Compute session / match impulse executed on this day
            if fixture and fixture.is_match and plan.planned_minutes > 0:
                impulse = (plan.planned_minutes / 90.0) * 100.0 * (1.0 + (plan.session_intensity * 0.25))
            else:
                impulse = (plan.session_intensity * 40.0) + (plan.target_high_speed_running_m / 15.0)

            # 3. Apply impulse & decay forward to next morning
            curr_fitness = curr_fitness * np.exp(-1.0 / self.tau_fitness) + (self.k_fitness * impulse * 0.08)
            curr_fatigue = curr_fatigue * np.exp(-1.0 / self.tau_fatigue) + (self.k_fatigue * impulse * 0.22)

        return snapshots

    def compare_scenarios(
        self,
        baseline_fitness: float,
        baseline_fatigue: float,
        schedule: List[ScheduledFixture],
        scenarios: Dict[str, List[ProposedAction]],
        tissue_vulnerability_index: float = 1.0
    ) -> Dict[str, List[DayTrajectorySnapshot]]:
        """
        Runs multiple coaching plans side-by-side to compare trade-offs.
        Example: 'Full Play' vs 'Subbed at 60' vs 'Benched & Primed'.
        """
        comparison = {}
        for name, plan in scenarios.items():
            comparison[name] = self.simulate_trajectory(
                baseline_fitness=baseline_fitness,
                baseline_fatigue=baseline_fatigue,
                schedule=schedule,
                coaching_plan=plan,
                tissue_vulnerability_index=tissue_vulnerability_index
            )
        return comparison

    def _evaluate_tactical_decision(
        self,
        fixture: ScheduledFixture | None,
        plan: ProposedAction,
        calibrated_risk: float
    ) -> Tuple[str, str]:
        """Maps physical risk and fixture leverage into actionable decisions."""
        if not fixture or not fixture.is_match:
            return ("TRAINING_DAY", "Standard microcycle session.")

        if calibrated_risk >= 0.40:
            return ("CLINICAL_REST", "Severe soft-tissue risk. Immediate medical rest required.")

        # High stakes match (Tier 1: Finals, Derbies, Knockouts)
        if fixture.leverage_tier == MatchLeverage.TIER_1_MUST_WIN:
            if calibrated_risk > 0.15:
                return (
                    "START_WITH_MINUTE_CAP_60",
                    f"Tier 1 match against {fixture.opponent}. High stakes justify start, but cap at 60 mins to prevent late eccentric breakdown."
                )
            return ("CLEARED_FULL_MATCH", f"Optimal condition for Tier 1 fixture against {fixture.opponent}.")

        # Routine or Rotation Match (Tier 2 / Tier 3)
        if calibrated_risk > 0.14:
            return (
                "BENCH_AND_PRESCRIBE_MICRODOSE",
                f"Elevated risk ({calibrated_risk*100:.1f}%) for routine fixture against {fixture.opponent}. Rest from match and prescribe 300m HSR priming session."
            )

        return ("CLEARED_FULL_MATCH", f"Cleared for match against {fixture.opponent}.")
