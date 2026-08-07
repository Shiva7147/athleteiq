"""Evidence-Backed RAG and Scientific Citation Retrieval Engine.

Retrieves peer-reviewed sports science citations based on query intent and athlete
deterministic risk metrics to synthesize structured clinical recommendations.
"""

import re
from typing import List
from src.ai.knowledge_base import SPORTS_SCIENCE_CORPUS
from src.domain.schemas import Citation, RiskAssessmentRead, StructuredDecisionResponse, WorkloadSummaryRead


def retrieve_relevant_citations(query_text: str, risk_assessment: RiskAssessmentRead) -> List[Citation]:
    """Retrieves relevant academic citations based on query keywords and athlete risk tier.

    Args:
        query_text: Coach's natural language question.
        risk_assessment: Deterministic `RiskAssessmentRead` metrics.

    Returns:
        List of relevant `Citation` objects.
    """
    selected: List[Citation] = []
    query_lower = query_text.lower()

    # Match based on risk tier and query keywords
    for citation in SPORTS_SCIENCE_CORPUS:
        score = 0
        citation_text = (citation.title + " " + citation.key_finding + " " + citation.journal).lower()

        # Keyword matching
        keywords = re.findall(r"\w+", query_lower)
        for kw in keywords:
            if len(kw) > 3 and kw in citation_text:
                score += 1

        # Domain contextual boosting
        if risk_assessment.risk_tier in ["HIGH", "CRITICAL"] and ("injury" in citation_text or "acwr" in citation_text or "spike" in citation_text):
            score += 3
        if "sleep" in query_lower and "sleep" in citation_text:
            score += 3
        if "hrv" in query_lower and "hrv" in citation_text:
            score += 3

        if score > 0:
            selected.append((score, citation))

    # Sort by relevance score descending
    selected.sort(key=lambda item: item[0], reverse=True)

    if not selected:
        # Fallback to top 2 foundational citations
        return SPORTS_SCIENCE_CORPUS[:2]

    return [item[1] for item in selected[:3]]


def generate_structured_decision(
    athlete_id: str,
    query_text: str,
    workload: WorkloadSummaryRead,
    risk: RiskAssessmentRead,
) -> StructuredDecisionResponse:
    """Synthesizes deterministic workload/risk calculations with scientific citations.

    Args:
        athlete_id: Target athlete ID.
        query_text: Coach's query text.
        workload: Deterministic `WorkloadSummaryRead`.
        risk: Deterministic `RiskAssessmentRead`.

    Returns:
        `StructuredDecisionResponse` DTO.
    """
    citations = retrieve_relevant_citations(query_text, risk)

    # Generate tailored action points based on deterministic risk tier
    action_points: List[str] = []
    if risk.risk_tier == "CRITICAL":
        action_points = [
            f"Enforce 24-48h active recovery protocol due to EWMA ACWR of {workload.acwr_ewma} (Danger Zone > 1.50).",
            "Reduce High-Speed Running (HSR) distance by 50% for the next 3 microcycles.",
            "Schedule manual therapy / physiotherapy screening for lower limb tightness.",
            "Prescribe sleep hygiene protocol aiming for +60-90 mins nocturnal sleep extension.",
        ]
        summary = (
            f"ATHLETE AT CRITICAL INJURY RISK (Score: {risk.risk_score}). "
            f"Acute training load has spiked severely relative to chronic fitness baseline. "
            f"Immediate load reduction is required to prevent acute soft-tissue strain."
        )
    elif risk.risk_tier == "HIGH":
        action_points = [
            f"Cap training session duration to 45 minutes (Current Monotony = {workload.monotony}).",
            "Reduce high-intensity sprint volume by 35%.",
            "Implement post-session cold-water immersion or active hydrotherapy.",
            "Re-evaluate HRV and RHR prior to next high-velocity field session.",
        ]
        summary = (
            f"ELEVATED INJURY RISK DETECTED (Score: {risk.risk_score}). "
            f"Athlete is experiencing elevated fatigue (EWMA ACWR = {workload.acwr_ewma}). "
            f"Moderating training volume is strongly advised."
        )
    elif risk.risk_tier == "MODERATE":
        action_points = [
            f"Maintain planned training volume but monitor in-session RPE.",
            "Ensure post-session nutrition and hydration within 30 minutes.",
            "Review sleep duration to close current sleep deficit.",
        ]
        summary = (
            f"MODERATE WORKLOAD TENSION (Score: {risk.risk_score}). "
            f"Workload is within acceptable boundaries (EWMA ACWR = {workload.acwr_ewma}), "
            f"but minor fatigue markers require routine monitoring."
        )
    else:
        action_points = [
            "Athlete is cleared for full planned training intensity and volume.",
            f"Current EWMA ACWR of {workload.acwr_ewma} is in the optimal 'Sweet Spot' (0.80 - 1.30).",
            "Continue progressive overload according to periodization schedule.",
        ]
        summary = (
            f"OPTIMAL WORKLOAD ADAPTATION (Score: {risk.risk_score}). "
            f"Athlete is well-conditioned with low injury risk. Maintain current periodization schedule."
        )

    return StructuredDecisionResponse(
        athlete_id=athlete_id,
        query_text=query_text,
        deterministic_acwr=workload.acwr_ewma,
        deterministic_risk_tier=risk.risk_tier,
        summary_recommendation=summary,
        action_points=action_points,
        citations=citations,
    )
