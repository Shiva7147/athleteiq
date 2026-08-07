"""Integration tests for Dense Vector Embedding and Vector Store Retrieval Engine."""

from datetime import date
from src.ai.embeddings import EmbeddingPipeline
from src.ai.vector_store import SportsScienceVectorStore
from src.models import InjuryRiskAssessmentRead, RiskTier


def test_embedding_pipeline_dimension() -> None:
    """Verifies dense embedding pipeline outputs 384-dimensional vectors."""
    pipeline = EmbeddingPipeline()
    vector = pipeline.embed_text("ACWR workload spikes and soft-tissue injury risk.")
    assert len(vector) == 384


def test_vector_store_similarity_search() -> None:
    """Verifies vector store querying top nearest neighbor research citations."""
    store = SportsScienceVectorStore()
    citations = store.similarity_search("What is the sweet spot for ACWR workload ratio?", top_k=2)
    assert len(citations) == 2
    assert any("Gabbett" in c.authors for c in citations)


def test_vector_store_contextual_risk_boosting() -> None:
    """Verifies contextual score boosting for critical risk assessment."""
    store = SportsScienceVectorStore()
    critical_risk = InjuryRiskAssessmentRead(
        athlete_id="ATH-100",
        assessment_date=date(2026, 8, 7),
        risk_tier=RiskTier.CRITICAL,
        risk_score=0.85,
        contributing_factors=["Critical EWMA Spike (1.65)"],
        recommended_action="Mandate active recovery.",
    )
    citations = store.similarity_search("High velocity training overload", risk_assessment=critical_risk, top_k=3)
    assert len(citations) == 3
    assert any("Gabbett" in c.authors or "Hulin" in c.authors for c in citations)
