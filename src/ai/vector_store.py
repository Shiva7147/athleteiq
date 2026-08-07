"""Sports Science Vector Store and Similarity Search Manager.

Manages vector collections over peer-reviewed sports science literature
using cosine similarity over 384-dimensional dense text embeddings.
"""

from typing import List, Optional, Tuple
import numpy as np

from src.ai.embeddings import EmbeddingPipeline
from src.ai.knowledge_base import SPORTS_SCIENCE_CORPUS
from src.models import Citation, InjuryRiskAssessmentRead


class SportsScienceVectorStore:
    """Vector database index for sports science literature retrieval."""

    def __init__(self, embedding_pipeline: Optional[EmbeddingPipeline] = None) -> None:
        self.embedder = embedding_pipeline or EmbeddingPipeline()
        self.corpus: List[Citation] = SPORTS_SCIENCE_CORPUS
        self.vectors: List[List[float]] = []
        self._initialize_index()

    def _initialize_index(self) -> None:
        """Encodes all research paper key findings into vector space."""
        doc_texts = [
            f"{c.title} {c.key_finding} {c.authors} {c.journal}"
            for c in self.corpus
        ]
        self.vectors = self.embedder.embed_documents(doc_texts)

    def similarity_search(
        self,
        query: str,
        risk_assessment: Optional[InjuryRiskAssessmentRead] = None,
        top_k: int = 3,
    ) -> List[Citation]:
        """Queries vector index for top-k nearest neighbor research citations.

        Args:
            query: Coach's query text.
            risk_assessment: Optional `InjuryRiskAssessmentRead` for contextual boosting.
            top_k: Number of citations to return.

        Returns:
            List of `Citation` objects.
        """
        query_vec = np.array(self.embedder.embed_text(query), dtype=np.float32)
        norm_q = np.linalg.norm(query_vec)
        if norm_q > 1e-6:
            query_vec /= norm_q

        scores: List[Tuple[float, Citation]] = []

        for idx, doc_vec in enumerate(self.vectors):
            d_vec = np.array(doc_vec, dtype=np.float32)
            norm_d = np.linalg.norm(d_vec)
            if norm_d > 1e-6:
                d_vec /= norm_d

            # Cosine similarity calculation
            similarity = float(np.dot(query_vec, d_vec))

            # Contextual boosting for high risk tiers
            if risk_assessment and risk_assessment.risk_tier in ["HIGH", "CRITICAL"]:
                citation = self.corpus[idx]
                if "injury" in citation.key_finding.lower() or "acwr" in citation.key_finding.lower():
                    similarity += 0.25

            scores.append((similarity, self.corpus[idx]))

        scores.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scores[:top_k]]
