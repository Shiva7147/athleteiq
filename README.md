# 🏃‍♂️ AthleteIQ Pro — AI Sports Science Decision Support Platform

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2.0-e91e63.svg)](https://docs.pydantic.dev/)
[![Pytest](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](https://docs.pytest.org/)

**AthleteIQ Pro** is an enterprise-grade AI Decision Support Platform built for sports science teams, medical staff, and high-performance coaches. It combines **pure deterministic mathematical algorithms** (sRPE, EWMA ACWR, Foster's Monotony & Strain) with **dense vector RAG search** over peer-reviewed sports medicine literature to generate actionable, evidence-backed recommendations without AI math hallucinations.

---

## 🏛️ System Architecture

```
                                  SYSTEM ARCHITECTURE

┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                Streamlit SaaS UI (src/ui)                                   │
│            • Squad Morning Availability Matrix    • What-If Training Load Simulator          │
│            • Athlete Workload Analytics           • Wearable CSV Data Ingester               │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │ HTTP REST
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  FastAPI Backend (src/api)                                  │
│             /api/v1/athletes • /api/v1/telemetry • /api/v1/analytics                       │
│             /api/v1/risk     • /api/v1/decisions/query                                      │
└──────────────┬───────────────────────────────┬───────────────────────────────┬──────────────┘
               │                               │                               │
               ▼                               ▼                               ▼
┌──────────────────────────────┐┌──────────────────────────────┐┌──────────────────────────────┐
│  Deterministic Analytics     ││   Calibrated Risk Engine     ││   Dense Vector RAG Engine    │
│    (src/analytics/acwr)      ││   (src/services/risk_service)││     (src/ai/vector_store)    │
│  • sRPE • Acute/Chronic Load ││  • Physiological Z-Scores    ││  • SentenceTransformers      │
│  • Uncoupled & EWMA ACWR     ││  • Non-Linear Risk Thresholds││  • Dense 384d Cosine Search  │
│  • Monotony & Strain Index   ││  • Factor Attributions       ││  • Peer-Reviewed Citations   │
└──────────────┬───────────────┘└──────────────┬───────────────┘└──────────────┬───────────────┘
               │                               │                               │
               └───────────────────────────────┼───────────────────────────────┘
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                          SQLAlchemy ORM & Data Repository Layer                             │
│                  Composite Index idx_telemetry_athlete_date on (athlete_id, recorded_date)  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Key Features

- 🚨 **Squad Morning Availability Matrix**: Team-wide Red/Amber/Green readiness table showing injury risk probability across the squad.
- 🔮 **Interactive "What-If" Training Load Simulator**: Models proposed session duration and RPE to project match-day EWMA ACWR spikes before prescribing field sessions.
- 🧮 **Pure Deterministic Analytics Engine**: $O(N)$ high-throughput formulas for sRPE, 7-day Acute Load, 28-day Chronic Load, Uncoupled ACWR, EWMA ACWR ($\lambda = 2/(N+1)$), Foster's Monotony, and Foster's Strain.
- 🤖 **Evidence-Backed Vector RAG**: Dense vector embeddings (`sentence-transformers/all-MiniLM-L6-v2`) querying peer-reviewed sports medicine literature (Gabbett 2016, Hulin 2014, Foster 1998, Blanch & Gabbett 2016, Impellizzeri 2019, Meeusen 2013).
- 📁 **Wearable CSV Drag-and-Drop Ingester**: Parsers for Catapult GPS, WHOOP, and Garmin export logs.
- 🛠️ **FastAPI Production REST API**: 8 enterprise endpoints with OpenAPI Swagger documentation and global error handlers.

---

## 🛠️ Technology Stack

- **Core**: Python 3.12+
- **API Framework**: FastAPI, Uvicorn
- **Data Validation & DTOs**: Pydantic v2
- **Database & Persistence**: SQLite / PostgreSQL, SQLAlchemy 2.0 ORM
- **AI Vector Search & Embeddings**: SentenceTransformers, ChromaDB
- **Frontend SaaS UI**: Streamlit, Plotly Express
- **Test Automation**: Pytest

---

## 🚀 Quick Start Guide

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Shiva7147/athleteiq.git
cd athleteiq
pip install -r pyproject.toml
```

### 2. Run Streamlit SaaS Dashboard

Launch the interactive web platform:

```bash
streamlit run src/ui/app.py
```
Open **`http://localhost:8501`** in your browser. Click **"🛠️ Demo Data Seeder"** to seed a 30-day squad telemetry dataset!

### 3. Run FastAPI Production REST API

Launch the backend API service:

```bash
uvicorn src.api.main:app --port 8000 --reload
```
Interactive OpenAPI Swagger UI: **`http://localhost:8000/docs`**

---

## 🧪 Running Unit & Integration Tests

Run the complete Pytest test suite:

```bash
pytest tests/unit/ tests/integration/ -v
```

---

## 📄 License

MIT License. Developed for high-performance sports science teams.
