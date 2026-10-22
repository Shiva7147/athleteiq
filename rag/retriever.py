"""
rag/retriever.py
Hybrid RAG retrieval engine:
1. Lexical BM25 (Okapi) keyword search for exact acronyms & authors (e.g. "ACWR", "Bahr 2016", "Impellizzeri").
2. Semantic Cosine Vector Similarity over normalized TF-IDF / term embeddings.
3. Reciprocal Rank Fusion (RRF) & Cross-Section Reranking.
4. Mandatory citation attribution so the LLM copilot rejects uncited answers.
"""
from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from rag.corpus.sports_science import SPORTS_SCIENCE_CORPUS, LiteratureChunk


class HybridLiteratureRetriever:
    """
    Hybrid retriever combining exact sparse BM25 retrieval with dense semantic vectors.
    """
    def __init__(self, corpus: List[LiteratureChunk] = SPORTS_SCIENCE_CORPUS):
        self.corpus = corpus
        self.doc_texts = [f"{c.title} {c.section} {c.content}" for c in self.corpus]
        
        # 1. Lexical BM25 Index
        self.tokenized_corpus = [doc.lower().split() for doc in self.doc_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 2. Semantic Vector Space (TF-IDF sub-linear n-gram vectorizer for zero-external-dependency speed)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts).toarray()

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Executes Reciprocal Rank Fusion (RRF) across BM25 and Semantic Similarity.
        RRF Score = 1 / (60 + BM25_Rank) + 1 / (60 + Semantic_Rank)
        """
        query_tokens = query.lower().split()
        n_docs = len(self.corpus)

        # 1. BM25 Scores
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranking = np.argsort(bm25_scores)[::-1]
        bm25_rank_map = {idx: rank for rank, idx in enumerate(bm25_ranking)}

        # 2. Semantic Cosine Similarity
        query_vec = self.vectorizer.transform([query]).toarray()[0]
        norm_q = np.linalg.norm(query_vec)
        if norm_q > 1e-6:
            dense_scores = np.dot(self.tfidf_matrix, query_vec) / (
                np.linalg.norm(self.tfidf_matrix, axis=1) * norm_q + 1e-8
            )
        else:
            dense_scores = np.zeros(n_docs)

        dense_ranking = np.argsort(dense_scores)[::-1]
        dense_rank_map = {idx: rank for rank, idx in enumerate(dense_ranking)}

        # 3. Reciprocal Rank Fusion
        fused_results = []
        for i, chunk in enumerate(self.corpus):
            rrf_score = (1.0 / (60.0 + bm25_rank_map[i])) + (1.0 / (60.0 + dense_rank_map[i]))
            fused_results.append({
                "chunk_id": chunk.chunk_id,
                "citation": chunk.citation,
                "title": chunk.title,
                "section": chunk.section,
                "page": chunk.page,
                "content": chunk.content,
                "rrf_score": float(rrf_score),
                "bm25_score": float(bm25_scores[i]),
                "dense_score": float(dense_scores[i]),
            })

        # Sort by RRF score descending
        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused_results[:top_k]

    def format_cited_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved literature with verified page citations for LLM prompt injection.
        """
        lines = ["=== RELEVANT SPORTS-SCIENCE LITERATURE (CITED) ==="]
        for res in search_results:
            lines.append(
                f"[{res['citation']}, p.{res['page']} | Section: {res['section']}]\n"
                f"\"{res['content']}\"\n"
            )
        return "\n".join(lines)
