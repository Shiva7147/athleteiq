"""
tests/test_rag.py
Unit tests verifying hybrid BM25 + dense retrieval, reciprocal rank fusion,
and literature citation formatting.
"""
import pytest
from rag.retriever import HybridLiteratureRetriever


def test_retriever_bm25_acronym_lookup():
    """Queries mentioning 'ACWR' must return Impellizzeri et al. as top result."""
    retriever = HybridLiteratureRetriever()
    results = retriever.search("What are the mathematical pitfalls of ACWR?", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "impellizzeri_2020_acwr_pitfalls"
    assert "Impellizzeri" in results[0]["citation"]
    assert results[0]["page"] == 909


def test_retriever_semantic_injury_prediction():
    """Queries regarding the impossibility of individual injury prediction must return Bahr 2016."""
    retriever = HybridLiteratureRetriever()
    results = retriever.search("Can we reliably predict individual soft tissue injuries?", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "bahr_2016_screening_limits"
    assert "Bahr R." in results[0]["citation"]


def test_retriever_sprint_microdosing():
    """Queries regarding rest decapacitation must return Gabbett 2016 training injury paradox."""
    retriever = HybridLiteratureRetriever()
    results = retriever.search("Why should we prescribe high-speed micro-dosing instead of full rest?", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "gabbett_2016_training_injury_paradox"
    assert "Gabbett TJ" in results[0]["citation"]


def test_cited_context_formatting():
    """Verifies that formatted prompt context includes explicit page numbers and citations."""
    retriever = HybridLiteratureRetriever()
    results = retriever.search("ACWR", top_k=1)
    formatted = retriever.format_cited_context(results)
    assert "=== RELEVANT SPORTS-SCIENCE LITERATURE (CITED) ===" in formatted
    assert "p.909" in formatted
    assert "Section: Methodological Pitfalls of ACWR" in formatted
