"""End-to-End Integration Tests for FastAPI v1 REST Endpoints."""

import os
from datetime import date, timedelta
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.dependencies import get_db_session
from src.api.main import app
from src.database.connection import Base

TEST_DB_PATH = "test_api_v1_athleteiq.db"
TEST_ENGINE = create_engine(f"sqlite:///{TEST_DB_PATH}", connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)


def override_get_db_session():
    """Override database dependency to use test SQLite database."""
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db_session] = override_get_db_session
client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db():
    """Reset database tables before each test execution."""
    Base.metadata.drop_all(bind=TEST_ENGINE)
    Base.metadata.create_all(bind=TEST_ENGINE)
    yield
    Base.metadata.drop_all(bind=TEST_ENGINE)
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except PermissionError:
            pass


def test_api_v1_health() -> None:
    """Verifies GET / health endpoint."""
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_api_v1_athletes_and_telemetry_flow() -> None:
    """Verifies athlete registration, telemetry logging, analytics, risk, and decision queries."""
    # 1. Create Athlete Profile
    res_athlete = client.post(
        "/api/v1/athletes",
        json={
            "name": "Christian Eriksen",
            "sport": "Football",
            "position": "Midfielder",
            "age": 31,
            "target_sleep_hours": 8.0,
        },
    )
    assert res_athlete.status_code == 201
    athlete_id = res_athlete.json()["id"]

    # 2. Log 30 days of telemetry
    start_date = date.today() - timedelta(days=30)
    for i in range(30):
        t_date = start_date + timedelta(days=i)
        t_payload = {
            "athlete_id": athlete_id,
            "recorded_date": t_date.isoformat(),
            "hr_rest_bpm": 50,
            "hrv_rmssd_ms": 70.0,
            "sleep_hours": 8.0,
            "rpe_score": 6.0,
            "session_duration_minutes": 60.0,
            "total_distance_meters": 8000.0,
            "high_speed_running_meters": 600.0,
        }
        res_t = client.post("/api/v1/telemetry", json=t_payload)
        assert res_t.status_code == 201

    # 3. GET Analytics
    res_analytics = client.get(f"/api/v1/analytics/{athlete_id}")
    assert res_analytics.status_code == 200
    assert res_analytics.json()["acwr_uncoupled"] == 1.0

    # 4. GET Risk Assessment
    res_risk = client.get(f"/api/v1/risk/{athlete_id}")
    assert res_risk.status_code == 200
    assert res_risk.json()["risk_tier"] == "LOW"

    # 5. POST Decision Query
    res_decision = client.post(
        "/api/v1/decisions/query",
        json={
            "athlete_id": athlete_id,
            "query_text": "Is this athlete ready for high-intensity match play?",
        },
    )
    assert res_decision.status_code == 200
    assert len(res_decision.json()["action_points"]) > 0
    assert len(res_decision.json()["citations"]) > 0
