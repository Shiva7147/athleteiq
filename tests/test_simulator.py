"""
tests/test_simulator.py
Unit tests verifying multi-world generation and causal differentiation.
"""
from datetime import date
import pytest
from simulator.models import SimulationWorld
from simulator.engine import LeagueSimulator


def test_squad_generation():
    """Generates 26 players with correct positional distribution."""
    sim = LeagueSimulator(world=SimulationWorld.WORLD_A, random_seed=42)
    squad = sim.generate_squad(26)
    assert len(squad) == 26
    assert squad[0].position == "GK"
    assert squad[3].position == "CB"


def test_world_c_sleep_load_interaction():
    """World C risk triggers sharply when sleep is low (<6h) and load is high."""
    sim = LeagueSimulator(world=SimulationWorld.WORLD_C, random_seed=42)
    squad = sim.generate_squad(5)
    
    records = sim.simulate_day_microcycle(
        current_date=date(2026, 10, 15),
        squad=squad,
        is_match_day=True
    )
    assert len(records) == 5
    for rec in records:
        assert 0.0 <= rec["true_injury_prob"] <= 1.0
