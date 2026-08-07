"""Unit tests for Pydantic application configuration settings."""

from src.config.settings import Settings, settings


def test_default_settings_instantiation() -> None:
    """Verifies default settings instantiation and database URL property."""
    s = Settings(db_path="test.db")
    assert s.app_name == "AthleteIQ Pro"
    assert s.database_url == "sqlite:///test.db"
    assert s.embedding_model_name == "all-MiniLM-L6-v2"


def test_global_settings_singleton() -> None:
    """Verifies global settings instance availability."""
    assert settings.app_name == "AthleteIQ Pro"
