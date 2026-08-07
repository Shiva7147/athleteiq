"""Unit tests for Evidence-Backed RAG and Scientific Citation Retrieval Engine."""

from datetime import date
from src.ai.rag_engine import generate_structured_decision, retrieve_relevant_citations
from src.domain.schemas import RiskAssessmentRead, RiskTier, WorkloadSummaryRead


def test_retrieve_relevant_citations() -> None:
    """Verifies citation matching for ACWR and sleep query terms."""
    risk = RiskAssessmentRead(
        athlete_id="ATH-100",
        assessment_date=date(2026, 8, 7),
        risk_tier=RiskTier.HIGH,
        risk_score=0.65,
        contributing_factors=["Critical EWMA Spike"],
        recommended_action="Reduce training duration.",
    )
    citations = retrieve_relevant_citations("What is the impact of EWMA workload spikes and sleep?", risk)
    assert len(citations) > 0
    assert any("Gabbett" in c.authors or "Foster" in c.authors for c in citations)


def test_generate_structured_decision() -> None:
    """Verifies generation of structured decision output with citations."""
    workload = WorkloadSummaryRead(
        athlete_id="ATH-100",
        evaluation_date=date(2026, 8, 7),
        acute_workload_7d=600.0,
        chronic_workload_28d=400.0,
        acwr_uncoupled=1.5,
        acwr_ewma=1.65,
        monotony=2.2,
        strain=4200.0,
        risk_zone="DANGER_ZONE",
    )
    risk = RiskAssessmentRead(
        athlete_id="ATH-100",
        assessment_date=date(2026, 8, 7),
        risk_tier=RiskTier.CRITICAL,
        risk_score=0.85,
        contributing_factors=["Critical Workload Spike (EWMA ACWR = 1.65)"],
        recommended_action="Enforce 24-48h active recovery.",
    )
    res = generate_structured_decision("ATH-100", "Should this athlete run high speed drills today?", workload, risk)
    assert res.athlete_id == "ATH-100"
    assert res.deterministic_risk_tier == RiskTier.CRITICAL
    assert len(res.action_points) >= 3
    assert len(res.citations) >= 1
