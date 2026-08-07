"""Streamlit Production SaaS Dashboard for AthleteIQ Pro.

Provides a modern, high-performance UI for coaches and sports scientists
to manage athletes, log session telemetry, view deterministic ACWR/workload charts,
and query the evidence-backed RAG decision engine.
"""

from datetime import date, timedelta
import pandas as pd
import streamlit as st

from src.db.database import SessionLocal, init_db
from src.db.repository import create_athlete, get_athlete, get_telemetry_history, list_athletes, log_telemetry
from src.domain.schemas import AthleteCreate, TelemetryCreate
from src.services.analytics import compute_workload_summary
from src.services.risk_engine import evaluate_risk
from src.ai.rag_engine import generate_structured_decision

# Initialize Database
init_db()

# Page configuration
st.set_page_config(
    page_title="AthleteIQ Pro | Sports Science Decision Platform",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #9CA3AF;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E293B;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">AthleteIQ Pro 🏃‍♂️⚡</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Decision Support Platform for High-Performance Sports Science</div>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select Module",
    [
        "👥 Athlete Roster",
        "📝 Log Session Telemetry",
        "📊 Workload Analytics & ACWR",
        "🤖 AI Decision Support & Citations",
        "🛠️ Seed Demo Dataset",
    ],
)


def get_db_session():
    """Helper to get a database session."""
    return SessionLocal()


# ----------------------------------------------------
# MODULE 1: ATHLETE ROSTER
# ----------------------------------------------------
if menu == "👥 Athlete Roster":
    st.header("Athlete Profile Management")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Register New Athlete Profile")
        with st.form("new_athlete_form"):
            name = st.text_input("Full Name", value="Alex Rivera")
            sport = st.selectbox("Sport", ["Football", "Basketball", "Track & Field", "Rugby", "Swimming"])
            position = st.text_input("Position / Specialty", value="Midfielder")
            age = st.number_input("Age", min_value=14, max_value=50, value=24)
            target_sleep = st.number_input("Target Sleep Hours", min_value=6.0, max_value=12.0, value=8.5, step=0.5)
            submitted = st.form_submit_button("Create Athlete Profile")

            if submitted:
                db = get_db_session()
                try:
                    new_athlete = create_athlete(
                        db,
                        AthleteCreate(
                            name=name,
                            sport=sport,
                            position=position,
                            age=age,
                            target_sleep_hours=target_sleep,
                        ),
                    )
                    st.success(f"Profile created successfully! Assigned ID: `{new_athlete.id}`")
                finally:
                    db.close()

    with col2:
        st.subheader("Current Roster")
        db = get_db_session()
        try:
            athletes = list_athletes(db)
            if athletes:
                df_athletes = pd.DataFrame([a.model_dump() for a in athletes])
                st.dataframe(df_athletes[["id", "name", "sport", "position", "age", "target_sleep_hours"]], use_container_width=True)
            else:
                st.info("No athletes registered yet. Use the form to create a profile or seed demo data.")
        finally:
            db.close()


# ----------------------------------------------------
# MODULE 2: LOG SESSION TELEMETRY
# ----------------------------------------------------
elif menu == "📝 Log Session Telemetry":
    st.header("Log Daily Wearable & Session Telemetry")
    db = get_db_session()
    try:
        athletes = list_athletes(db)
        if not athletes:
            st.warning("Please register an athlete profile first before logging telemetry.")
        else:
            athlete_dict = {f"{a.name} ({a.id})": a.id for a in athletes}
            selected_label = st.selectbox("Select Athlete", list(athlete_dict.keys()))
            athlete_id = athlete_dict[selected_label]

            with st.form("log_telemetry_form"):
                col_a, col_b = st.columns(2)
                with col_a:
                    recorded_date = st.date_input("Date", value=date.today())
                    rpe = st.slider("Session RPE (Borg Scale 1-10)", min_value=1.0, max_value=10.0, value=7.0, step=0.5)
                    duration = st.number_input("Session Duration (minutes)", min_value=0.0, max_value=300.0, value=90.0, step=5.0)
                    rhr = st.number_input("Resting HR (bpm)", min_value=30, max_value=200, value=52)
                with col_b:
                    hrv = st.number_input("HRV rMSSD (ms)", min_value=5.0, max_value=250.0, value=65.0, step=1.0)
                    sleep = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=16.0, value=7.5, step=0.5)
                    total_dist = st.number_input("Total GPS Distance (meters)", min_value=0.0, max_value=20000.0, value=7500.0, step=100.0)
                    hsr_dist = st.number_input("High Speed Running >19.8 km/h (meters)", min_value=0.0, max_value=5000.0, value=650.0, step=50.0)
                    injuries = st.text_input("Notes / Subjective Soreness", value="")

                submit_log = st.form_submit_button("Submit Telemetry Log")

                if submit_log:
                    t_entry = TelemetryCreate(
                        athlete_id=athlete_id,
                        recorded_date=recorded_date,
                        hr_rest_bpm=rhr,
                        hrv_rmssd_ms=hrv,
                        sleep_hours=sleep,
                        rpe_score=rpe,
                        session_duration_minutes=duration,
                        total_distance_meters=total_dist,
                        high_speed_running_meters=hsr_dist,
                        injuries_reported=injuries if injuries else None,
                    )
                    log_telemetry(db, t_entry)
                    st.success(f"Telemetry recorded for {recorded_date}! Session Load = {rpe * duration:.1f} AU")

            st.subheader("Recent Telemetry History")
            history = get_telemetry_history(db, athlete_id)
            if history:
                df_history = pd.DataFrame([h.model_dump() for h in history])
                st.dataframe(df_history[["recorded_date", "rpe_score", "session_duration_minutes", "session_load", "hr_rest_bpm", "hrv_rmssd_ms", "sleep_hours"]].tail(10), use_container_width=True)
    finally:
        db.close()


# ----------------------------------------------------
# MODULE 3: WORKLOAD ANALYTICS & ACWR
# ----------------------------------------------------
elif menu == "📊 Workload Analytics & ACWR":
    st.header("Deterministic Workload & ACWR Analytics")
    db = get_db_session()
    try:
        athletes = list_athletes(db)
        if not athletes:
            st.warning("No athletes registered.")
        else:
            athlete_dict = {f"{a.name} ({a.id})": a.id for a in athletes}
            selected_label = st.selectbox("Select Athlete for Analytics", list(athlete_dict.keys()))
            athlete_id = athlete_dict[selected_label]

            history = get_telemetry_history(db, athlete_id)
            if len(history) < 28:
                st.warning(f"ACWR & Deterministic Workload Analytics require at least 28 days of telemetry history. Currently found **{len(history)}** records. Use the 'Seed Demo Dataset' tab to generate 30 days of data instantly.")
            else:
                workload = compute_workload_summary(history)
                athlete_obj = get_athlete(db, athlete_id)
                risk = evaluate_risk(history, target_sleep_hours=athlete_obj.target_sleep_hours)

                # Metric Cards
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("EWMA ACWR", f"{workload.acwr_ewma}", delta=workload.risk_zone)
                c2.metric("Uncoupled ACWR", f"{workload.acwr_uncoupled}")
                c3.metric("Acute Workload (7d)", f"{workload.acute_workload_7d:.0f} AU")
                c4.metric("Chronic Workload (28d)", f"{workload.chronic_workload_28d:.0f} AU")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Training Monotony", f"{workload.monotony}")
                c6.metric("Training Strain", f"{workload.strain:.0f} AU")
                c7.metric("Risk Score", f"{risk.risk_score * 100:.1f}%")
                c8.metric("Risk Tier", f"{risk.risk_tier.value}")

                st.subheader("Workload & ACWR Rolling Trend")
                df_h = pd.DataFrame([h.model_dump() for h in history])
                st.line_chart(df_h.set_index("recorded_date")[["session_load", "hrv_rmssd_ms"]])
    finally:
        db.close()


# ----------------------------------------------------
# MODULE 4: AI DECISION SUPPORT & CITATIONS
# ----------------------------------------------------
elif menu == "🤖 AI Decision Support & Citations":
    st.header("Evidence-Backed RAG Decision Support Engine")
    st.caption("Combines deterministic calculations with peer-reviewed sports science literature.")

    db = get_db_session()
    try:
        athletes = list_athletes(db)
        if not athletes:
            st.warning("Please register an athlete and seed data first.")
        else:
            athlete_dict = {f"{a.name} ({a.id})": a.id for a in athletes}
            selected_label = st.selectbox("Select Target Athlete", list(athlete_dict.keys()))
            athlete_id = athlete_dict[selected_label]

            history = get_telemetry_history(db, athlete_id)
            if len(history) < 28:
                st.warning("Requires at least 28 days of history to generate decision support.")
            else:
                user_query = st.text_area(
                    "Ask a Sports Performance or Workload Question",
                    value="Should this athlete participate in high-speed running drills today given their current EWMA spike and HRV?",
                )

                if st.button("Generate Evidence-Backed Recommendation"):
                    athlete_obj = get_athlete(db, athlete_id)
                    workload = compute_workload_summary(history)
                    risk = evaluate_risk(history, target_sleep_hours=athlete_obj.target_sleep_hours)
                    response = generate_structured_decision(athlete_id, user_query, workload, risk)

                    st.subheader("📋 Executive Recommendation")
                    st.info(response.summary_recommendation)

                    st.subheader("🎯 Tactical Action Points")
                    for ap in response.action_points:
                        st.markdown(f"- {ap}")

                    st.subheader("📚 Peer-Reviewed Scientific Citations")
                    for cit in response.citations:
                        with st.expander(f"📖 {cit.authors} ({cit.year}) — {cit.title}"):
                            st.write(f"**Journal**: {cit.journal}")
                            st.write(f"**Key Finding**: {cit.key_finding}")
                            st.markdown(f"[View Publication]({cit.doi_or_link})")
    finally:
        db.close()


# ----------------------------------------------------
# MODULE 5: SEED DEMO DATASET
# ----------------------------------------------------
elif menu == "🛠️ Seed Demo Dataset":
    st.header("Seed Synthetic 30-Day Telemetry Dataset")
    st.write("Generates 30 days of realistic daily training loads, resting heart rate, HRV, and sleep metrics for instant evaluation.")

    if st.button("Seed Demo Athlete & 30-Day Telemetry"):
        db = get_db_session()
        try:
            demo_athlete = create_athlete(
                db,
                AthleteCreate(
                    name="Marcus Vance",
                    sport="Football",
                    position="Winger",
                    age=23,
                    target_sleep_hours=8.0,
                ),
            )
            athlete_id = demo_athlete.id

            start_date = date.today() - timedelta(days=30)
            for i in range(30):
                curr_date = start_date + timedelta(days=i)
                # Introduce a spike in the last 5 days
                rpe_val = 8.5 if i >= 25 else 5.5 + (i % 3) * 0.5
                duration_val = 90.0 if i >= 25 else 60.0 + (i % 2) * 15.0
                hrv_val = 45.0 if i >= 25 else 68.0 - (i % 4)
                rhr_val = 62 if i >= 25 else 52 + (i % 3)

                t = TelemetryCreate(
                    athlete_id=athlete_id,
                    recorded_date=curr_date,
                    hr_rest_bpm=rhr_val,
                    hrv_rmssd_ms=hrv_val,
                    sleep_hours=6.5 if i >= 25 else 8.0,
                    rpe_score=rpe_val,
                    session_duration_minutes=duration_val,
                    total_distance_meters=8500.0,
                    high_speed_running_meters=1100.0 if i >= 25 else 450.0,
                )
                log_telemetry(db, t)

            st.success(f"Successfully seeded athlete **{demo_athlete.name}** (`{athlete_id}`) with 30 days of telemetry history!")
        finally:
            db.close()
