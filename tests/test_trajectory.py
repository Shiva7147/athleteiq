"""
tests/test_trajectory.py
Unit tests verifying multi-day forward trajectory simulations, Banister dynamics,
and match leverage branch comparisons.
"""
import pytest
from core.types import MatchLeverage
from simulator.trajectory import (
    TrajectorySimulationEngine,
    ScheduledFixture,
    ProposedAction,
)


def test_trajectory_fatigue_and_fitness_rollout():
    """Verifies that high match minutes increase acute fatigue faster than fitness."""
    engine = TrajectorySimulationEngine()
    schedule = [
        ScheduledFixture(day_offset=0, opponent="League Rival", leverage_tier=MatchLeverage.TIER_2_STANDARD),
        ScheduledFixture(day_offset=3, opponent="Title Decider", leverage_tier=MatchLeverage.TIER_1_MUST_WIN),
    ]
    # Coach plays athlete 90 mins today
    plan = [
        ProposedAction(day_offset=0, planned_minutes=90, session_intensity=1.0, target_high_speed_running_m=600.0),
        ProposedAction(day_offset=1, planned_minutes=0, session_intensity=0.2, target_high_speed_running_m=0.0),
        ProposedAction(day_offset=2, planned_minutes=0, session_intensity=0.3, target_high_speed_running_m=100.0),
        ProposedAction(day_offset=3, planned_minutes=90, session_intensity=1.0, target_high_speed_running_m=700.0),
    ]

    trajectory = engine.simulate_trajectory(
        baseline_fitness=70.0,
        baseline_fatigue=20.0,
        schedule=schedule,
        coaching_plan=plan
    )

    assert len(trajectory) == 4
    # Day 0 match must increase fatigue significantly on Day 1
    assert trajectory[1].projected_fatigue > trajectory[0].projected_fatigue
    # Conformal intervals must be well-formed (lower < risk < upper)
    for snap in trajectory:
        assert snap.risk_ci_lower_95 <= snap.calibrated_injury_risk <= snap.risk_ci_upper_95


def test_scenario_comparison_full_play_vs_benched():
    """
    Simulates the tactical decision:
    Scenario A: Plays 90 mins on Day 0 in standard league match.
    Scenario B: Benched on Day 0 with micro-dose session.
    Verifies that on Day 3 (Tier 1 Title Decider), Scenario B yields lower risk & higher readiness.
    """
    engine = TrajectorySimulationEngine()
    schedule = [
        ScheduledFixture(day_offset=0, opponent="Mid-Table FC", leverage_tier=MatchLeverage.TIER_2_STANDARD),
        ScheduledFixture(day_offset=3, opponent="Championship Final", leverage_tier=MatchLeverage.TIER_1_MUST_WIN),
    ]

    scenarios = {
        "Scenario_A_Full_Play": [
            ProposedAction(day_offset=0, planned_minutes=90, session_intensity=1.0, target_high_speed_running_m=650.0),
            ProposedAction(day_offset=1, planned_minutes=0, session_intensity=0.2, target_high_speed_running_m=0.0),
            ProposedAction(day_offset=2, planned_minutes=0, session_intensity=0.4, target_high_speed_running_m=150.0),
            ProposedAction(day_offset=3, planned_minutes=90, session_intensity=1.0, target_high_speed_running_m=650.0),
        ],
        "Scenario_B_Benched_And_Primed": [
            ProposedAction(day_offset=0, planned_minutes=0, session_intensity=0.3, target_high_speed_running_m=250.0),
            ProposedAction(day_offset=1, planned_minutes=0, session_intensity=0.2, target_high_speed_running_m=0.0),
            ProposedAction(day_offset=2, planned_minutes=0, session_intensity=0.4, target_high_speed_running_m=200.0),
            ProposedAction(day_offset=3, planned_minutes=90, session_intensity=1.0, target_high_speed_running_m=650.0),
        ],
    }

    comparison = engine.compare_scenarios(
        baseline_fitness=65.0,
        baseline_fatigue=25.0,
        schedule=schedule,
        scenarios=scenarios
    )

    day3_a = comparison["Scenario_A_Full_Play"][3]
    day3_b = comparison["Scenario_B_Benched_And_Primed"][3]

    # Athlete entering Day 3 via Scenario B must have higher readiness and lower fatigue
    assert day3_b.readiness_score > day3_a.readiness_score
    assert day3_b.projected_fatigue < day3_a.projected_fatigue
    assert day3_b.calibrated_injury_risk < day3_a.calibrated_injury_risk


def test_minute_capping_in_tier_1_fixture():
    """
    Verifies that when an athlete enters a Tier 1 match with elevated risk (e.g. 20%),
    the engine recommends START_WITH_MINUTE_CAP_60 rather than outright benching.
    """
    engine = TrajectorySimulationEngine()
    schedule = [
        ScheduledFixture(day_offset=0, opponent="Champions League Final", leverage_tier=MatchLeverage.TIER_1_MUST_WIN)
    ]
    # Enter with high baseline fatigue
    plan = [
        ProposedAction(day_offset=0, planned_minutes=90, session_intensity=1.0, target_high_speed_running_m=500.0)
    ]

    trajectory = engine.simulate_trajectory(
        baseline_fitness=60.0,
        baseline_fatigue=55.0,  # High acute fatigue
        schedule=schedule,
        coaching_plan=plan
    )

    assert trajectory[0].decision_recommendation == "START_WITH_MINUTE_CAP_60"
    assert "cap at 60 mins" in trajectory[0].rationale
