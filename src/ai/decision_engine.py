"""Evidence-Backed Vector RAG Decision Support Engine.

Combines vector similarity search over peer-reviewed sports science literature
with deterministic workload and risk metrics to synthesize structured clinical recommendations.
"""

from typing import List, Optional
from src.ai.vector_store import SportsScienceVectorStore
from src.models import (
    Citation,
    InjuryRiskAssessmentRead,
    StructuredDecisionResponse,
    WorkloadSummaryRead,
)


class RAGDecisionEngine:
    """Vector-RAG Decision Engine for high-performance sports science."""

    def __init__(self, vector_store: Optional[SportsScienceVectorStore] = None) -> None:
        self.vector_store = vector_store or SportsScienceVectorStore()

    def generate_decision(
        self,
        athlete_id: str,
        query_text: str,
        workload: WorkloadSummaryRead,
        risk: InjuryRiskAssessmentRead,
    ) -> StructuredDecisionResponse:
        """Synthesizes deterministic workload/risk calculations with scientific vector citations.

        Args:
            athlete_id: Target athlete ID.
            query_text: Coach's query prompt.
            workload: Deterministic `WorkloadSummaryRead`.
            risk: Deterministic `InjuryRiskAssessmentRead`.

        Returns:
            `StructuredDecisionResponse` DTO.
        """
        citations = self.vector_store.similarity_search(query_text, risk_assessment=risk, top_k=3)

        action_points: List[str] = []
        if risk.risk_tier == "CRITICAL":
            action_points = [
                f"Enforce 24-48h active recovery due to EWMA ACWR of {workload.acwr_ewma} (Danger Zone > 1.50).",
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
                "Maintain planned training volume but monitor in-session RPE.",
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
