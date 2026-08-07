"""Centralized Configuration Settings for AthleteIQ Pro.

Uses Pydantic BaseSettings for strongly-typed environment configuration management.
"""

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Production application settings DTO."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field("AthleteIQ Pro", description="Application name")
    app_version: str = Field("1.0.0", description="Application version")
    environment: str = Field("production", description="Environment: development, testing, production")
    debug: bool = Field(False, description="Debug mode flag")

    # Database Settings
    db_path: str = Field("athleteiq.db", description="SQLite database file path")
    
    @property
    def database_url(self) -> str:
        """Returns SQLAlchemy database URL."""
        return f"sqlite:///{self.db_path}"

    # AI & Vector RAG Settings
    chroma_db_dir: str = Field("chroma_db", description="ChromaDB persistence directory")
    embedding_model_name: str = Field("all-MiniLM-L6-v2", description="SentenceTransformers embedding model")


# Global settings singleton instance
settings = Settings()
