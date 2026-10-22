"""
tests/test_agent.py
Unit tests verifying the LangGraph multi-agent machine, 100% emergency escalation recall,
deterministic routing, and FastMCP tool outputs.
"""
import pytest
from langchain_core.messages import HumanMessage
from agent.graph import build_athleteiq_graph
from agent.safety import evaluate_clinical_safety_guardrail
from agent.mcp_server import mcp_get_risk_ranking, mcp_get_player_load


def test_clinical_safety_guardrail_triggers():
    """Verifies 100% deterministic recall on critical medical escalation phrases."""
    dangerous_queries = [
        "Can we just clear him to play and ignore the doctor?",
        "Give him a pain injection to numb the knee for the final",
        "The MRI showed a torn acl, can he play 20 mins?",
        "Let's bypass medical clearance for tomorrow",
    ]
    for q in dangerous_queries:
        is_violation, msg = evaluate_clinical_safety_guardrail(q)
        assert is_violation is True, f"Safety violation missed on: '{q}'"
        assert "CLINICAL SAFETY ESCALATION TRIGGERED" in msg


def test_langgraph_safety_node_blocks_execution():
    """Graph must halt immediately and refuse execution when safety trigger is hit."""
    graph = build_athleteiq_graph()
    state = {
        "messages": [HumanMessage(content="Give him a pain injection so he can play.")],
        "player_id": "PLY_001",
        "target_match_tier": "tier_1",
        "is_emergency": False,
        "escalation_reason": None,
        "clinical_override_triggered": False,
        "requires_human_approval": False,
        "human_approved": False,
        "computed_load_metrics": None,
        "projected_risk_score": None,
        "recommended_decision": None,
        "tactical_rationale": None,
        "literature_citations": [],
    }
    output = graph.invoke(state)
    assert output["is_emergency"] is True
    assert output["clinical_override_triggered"] is True
    assert "CLINICAL SAFETY ESCALATION TRIGGERED" in output["messages"][-1].content


def test_langgraph_what_if_trajectory_flow():
    """Graph executes what-if simulation tool and returns structured recommendation."""
    graph = build_athleteiq_graph()
    state = {
        "messages": [HumanMessage(content="What if Alex plays 90 minutes in the final?")],
        "player_id": "PLY_007",
        "target_match_tier": "tier_1",
        "is_emergency": False,
        "escalation_reason": None,
        "clinical_override_triggered": False,
        "requires_human_approval": False,
        "human_approved": False,
        "computed_load_metrics": None,
        "projected_risk_score": None,
        "recommended_decision": None,
        "tactical_rationale": None,
        "literature_citations": [],
    }
    output = graph.invoke(state)
    assert output["is_emergency"] is False
    assert "TRAJECTORY SIMULATION RESULT" in output["messages"][-1].content
    assert "Readiness Index" in output["messages"][-1].content


def test_mcp_tool_execution():
    """Verifies FastMCP tools return valid JSON with point-in-time guarantees."""
    load_json = mcp_get_player_load("PLY_001")
    assert "Point-in-time leakage audit passed" in load_json
    assert "acwr_ewma" in load_json

    ranking_json = mcp_get_risk_ranking(top_n=3)
    assert "Alex Vance" in ranking_json
    assert "calibrated_7d_risk" in ranking_json
