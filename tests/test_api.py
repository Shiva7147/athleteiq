"""
tests/test_api.py
Integration tests for the AthleteIQ FastAPI REST gateway.
Verifies health checks, squad ranking, trajectory simulations, and agent endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from api.routes import app

client = TestClient(app)


def test_health_check_endpoint():
    """GET /v2/health must return 200 OK and status HEALTHY."""
    response = client.get("/v2/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["point_in_time_audit"] == "PASSED"
    assert "MonotonicLightGBM" in data["models_loaded"]


def test_serve_dashboard_endpoint():
    """GET / and /dashboard must return 200 OK and serve the HTML dashboard."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "AthleteIQ" in response.text
    assert "Clinical Decision Support Console" in response.text
    assert "Daily Squad Attention Triage" in response.text


def test_squad_ranking_endpoint():
    """GET /v2/squad/risk-ranking returns top athletes adhering to budget."""
    response = client.get("/v2/squad/risk-ranking?top_n=3")
    assert response.status_code == 200
    data = response.json()
    assert data["alert_budget_daily"] == 3
    assert len(data["ranked_athletes"]) == 3
    assert data["ranked_athletes"][0]["name"] == "Alex Vance (Winger)"


def test_simulate_trajectory_endpoint():
    """POST /v2/simulate/trajectory runs forward rollout."""
    payload = {
        "player_id": "PLY_007",
        "planned_minutes": 90,
        "is_tier_1_final": True
    }
    response = client.post("/v2/simulate/trajectory", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["player_id"] == "PLY_007"
    assert "readiness_score" in data
    assert "calibrated_injury_risk" in data


def test_morning_briefing_endpoint():
    """GET /v2/briefing/morning returns spoken script and storyboard scenes."""
    response = client.get("/v2/briefing/morning")
    assert response.status_code == 200
    data = response.json()
    assert "Alex Vance" in data["audio_script"]
    assert len(data["video_storyboard_scenes"]) == 4


def test_agent_chat_endpoint_safety_escalation():
    """POST /v2/agent/chat enforces safety override on unauthorized medical phrases."""
    payload = {
        "message": "Give him a pain injection to numb the knee for today",
        "player_id": "PLY_001"
    }
    response = client.post("/v2/agent/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_emergency"] is True
    assert data["clinical_override_triggered"] is True
    assert "CLINICAL SAFETY ESCALATION TRIGGERED" in data["reply"]
