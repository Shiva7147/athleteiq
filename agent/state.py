"""
agent/state.py
Strongly typed state definitions for the LangGraph multi-agent decision machine.
"""
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage


class AgentDecisionState(TypedDict):
    """
    State contract passed across nodes in the LangGraph decision pipeline.
    """
    messages: Sequence[BaseMessage]
    player_id: Optional[str]
    target_match_tier: Optional[str]
    is_emergency: bool
    escalation_reason: Optional[str]
    clinical_override_triggered: bool
    requires_human_approval: bool
    human_approved: bool
    computed_load_metrics: Optional[Dict[str, Any]]
    projected_risk_score: Optional[float]
    recommended_decision: Optional[str]
    tactical_rationale: Optional[str]
    literature_citations: List[str]
    tools_called: List[Dict[str, Any]]
