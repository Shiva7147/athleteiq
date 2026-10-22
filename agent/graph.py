"""
agent/graph.py
LangGraph state graph linking the Safety Gatekeeper, Tool Invocation, and Decision Router.
"""
from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from agent.state import AgentDecisionState
from agent.safety import evaluate_clinical_safety_guardrail
from agent.tools import get_player_load, get_risk_ranking, simulate_what_if_trajectory, search_literature


def safety_node(state: AgentDecisionState) -> Dict[str, Any]:
    """
    First gate: Pure deterministic code. Zero tokens sent to LLM.
    """
    last_msg = state["messages"][-1].content if state["messages"] else ""
    is_violation, escalation = evaluate_clinical_safety_guardrail(last_msg)
    
    if is_violation:
        return {
            "is_emergency": True,
            "escalation_reason": escalation,
            "clinical_override_triggered": True,
            "messages": list(state["messages"]) + [AIMessage(content=escalation)],
            "tools_called": [{"tool": "safety_guardrail", "status": "ESCALATED_REJECTED", "details": "Critical Medical Trigger Detected"}],
        }
        
    return {"is_emergency": False, "tools_called": []}


def route_safety(state: AgentDecisionState) -> str:
    if state.get("is_emergency", False):
        return "end_node"
    return "orchestrator_node"


def orchestrator_node(state: AgentDecisionState) -> Dict[str, Any]:
    """
    Processes the request by calling deterministic sports science tools.
    """
    last_msg = state["messages"][-1].content.lower() if state["messages"] else ""

    if "rank" in last_msg or "top 3" in last_msg or "morning briefing" in last_msg:
        ranking = get_risk_ranking(top_n=3)
        summary = (
            "MORNING SQUAD RISK RANKING (Alert Budget = 3):\n"
            f"1. {ranking[0]['name']} | 7d Risk: {ranking[0]['calibrated_7d_risk']*100:.1f}% | Action: {ranking[0]['recommended_action']}\n"
            f"   Driver: {ranking[0]['primary_shap_driver']}\n"
            f"2. {ranking[1]['name']} | 7d Risk: {ranking[1]['calibrated_7d_risk']*100:.1f}% | Action: {ranking[1]['recommended_action']}\n"
            f"   Driver: {ranking[1]['primary_shap_driver']}\n"
            f"3. {ranking[2]['name']} | 7d Risk: {ranking[2]['calibrated_7d_risk']*100:.1f}% | Action: {ranking[2]['recommended_action']}\n"
            f"   Driver: {ranking[2]['primary_shap_driver']}"
        )
        return {
            "messages": list(state["messages"]) + [AIMessage(content=summary)],
            "recommended_decision": ranking[0]["recommended_action"],
            "tools_called": [{"tool": "get_risk_ranking", "args": {"top_n": 3}, "status": "SUCCESS"}],
        }

    elif "what if" in last_msg or "simulate" in last_msg:
        # What-if scenario call
        is_final = "final" in last_msg or "cup" in last_msg
        res = simulate_what_if_trajectory("PLY_007", planned_minutes=90, is_tier_1_final=is_final)
        msg = (
            f"TRAJECTORY SIMULATION RESULT (Player: PLY_007, 90 mins):\n"
            f"- Projected Fatigue: {res['projected_fatigue']}\n"
            f"- Readiness Index: {res['readiness_score']} / 100\n"
            f"- Calibrated Injury Risk: {res['calibrated_injury_risk']*100:.1f}%\n"
            f"- Recommendation: {res['decision']}\n"
            f"- Tactical Rationale: {res['rationale']}"
        )
        return {
            "messages": list(state["messages"]) + [AIMessage(content=msg)],
            "recommended_decision": res["decision"],
            "tools_called": [{"tool": "simulate_what_if_trajectory", "args": {"player_id": "PLY_007", "minutes": 90, "is_final": is_final}, "status": "SUCCESS"}],
        }

    elif "literature" in last_msg or "acwr" in last_msg or "bahr" in last_msg or "gabbett" in last_msg:
        lit = search_literature(last_msg)
        top = lit[0] if lit else None
        if top:
            msg = (
                f"EVIDENCE SYNTHESIS:\n"
                f"Citation: [{top['citation']}, p.{top['page']}]\n"
                f"Finding: {top['content']}"
            )
        else:
            msg = "No directly relevant peer-reviewed papers found."
        return {
            "messages": list(state["messages"]) + [AIMessage(content=msg)],
            "literature_citations": [top["citation"]] if top else [],
            "tools_called": [{"tool": "search_literature", "args": {"query": last_msg[:40]}, "status": "SUCCESS", "citation": top["citation"] if top else None}],
        }

    else:
        # Default load lookup
        load = get_player_load("PLY_001")
        msg = (
            f"ATHLETE LOAD REPORT (PLY_001):\n"
            f"- Acute 7d EWMA: {load['acute_load_7d_ewma']}\n"
            f"- Chronic 28d EWMA: {load['chronic_load_28d_ewma']}\n"
            f"- ACWR (EWMA): {load['acwr_ewma']}\n"
            f"- Weekly Strain: {load['training_strain_7d']}\n"
            f"Verification: {load['verification']}"
        )
        return {
            "messages": list(state["messages"]) + [AIMessage(content=msg)],
            "tools_called": [{"tool": "get_player_load", "args": {"player_id": "PLY_001"}, "status": "SUCCESS"}],
        }


def build_athleteiq_graph():
    """
    Constructs the compiled LangGraph state workflow.
    """
    workflow = StateGraph(AgentDecisionState)
    workflow.add_node("safety_guard", safety_node)
    workflow.add_node("orchestrator", orchestrator_node)

    workflow.set_entry_point("safety_guard")
    workflow.add_conditional_edges(
        "safety_guard",
        route_safety,
        {
            "end_node": END,
            "orchestrator_node": "orchestrator",
        }
    )
    workflow.add_edge("orchestrator", END)
    return workflow.compile()
