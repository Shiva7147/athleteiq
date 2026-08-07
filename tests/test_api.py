"""Integration tests for FastAPI REST Application Endpoints."""

import os
from datetime import date, timedelta
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.database import Base, get_db
from src.api.main import app

TEST_DB_PATH = "test_api_athleteiq.db"
TEST_ENGINE = create_engine(f"sqlite:///{TEST_DB_PATH}", connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)


def override_get_db():
    """Override database dependency to use test SQLite database."""
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
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


def test_api_health_check() -> None:
    """Verifies GET / health endpoint."""
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_api_athlete_lifecycle() -> None:
    """Verifies creating, getting, and listing athletes via REST API."""
    # 1. Create athlete
    payload = {
        "name": "Marcus Vance",
        "sport": "Football",
        "position": "Winger",
        "age": 23,
        "target_sleep_hours": 8.0,
    }
    res_create = client.post("/api/v1/athletes", json=payload)
    assert res_create.status_code == 201
    athlete_id = res_create.json()["id"]

    # 2. Get athlete
    res_get = client.get(f"/api/v1/athletes/{athlete_id}")
    assert res_get.status_code == 200
    assert res_get.json()["name"] == "Marcus Vance"

    # 3. List athletes
    res_list = client.get("/api/v1/athletes")
    assert res_list.status_code == 200
    assert len(res_list.json()) == 1


def test_api_end_to_end_analytics_and_decisions() -> None:
    """Verifies logging 30 telemetry entries and calling analytics & decision endpoints."""
    # Create athlete
    res_create = client.post(
        "/api/v1/athletes",
        json={"name": "Leo Vance", "sport": "Rugby", "position": "Flanker", "age": 25},
    )
    athlete_id = res_create.json()["id"]

    # Log 30 days of telemetry
    start_date = date.today() - timedelta(days=30)
    for i in range(30):
        t_date = start_date + timedelta(days=i)
        t_payload = {
            "athlete_id": athlete_id,
            "recorded_date": t_date.isoformat(),
            "hr_rest_bpm": 52,
            "hrv_rmssd_ms": 65.0,
            "sleep_hours": 8.0,
            "rpe_score": 6.0,
            "session_duration_minutes": 60.0,
            "total_distance_meters": 7500.0,
            "high_speed_running_meters": 500.0,
        }
        res_t = client.post("/api/v1/telemetry", json=t_payload)
        assert res_t.status_code == 201

    # Call Analytics endpoint
    res_analytics = client.get(f"/api/v1/analytics/{athlete_id}")
    assert res_analytics.status_code == 200
    assert res_analytics.json()["acwr_uncoupled"] == 1.0

    # Call Risk endpoint
    res_risk = client.get(f"/api/v1/risk/{athlete_id}")
    assert res_risk.status_code == 200
    assert res_risk.json()["risk_tier"] == "LOW"

    # Call RAG Decision Query endpoint
    query_payload = {
        "athlete_id": athlete_id,
        "query_text": "Is this athlete ready for full match intensity?",
    }
    res_decision = client.post("/api/v1/decisions/query", json=query_payload)
    assert res_decision.status_code == 200
    assert len(res_decision.json()["action_points"]) > 0
    assert len(res_decision.json()["citations"]) > 0
