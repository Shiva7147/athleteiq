# tools.py
# AthleteIQ — Agent Tools
# Four tools the agent can call based on the coach's query

from langchain.tools import tool
from datetime import datetime

# Shared athlete data log — persists across tool calls
athlete_data_log = []

@tool
def search_sports_science(query: str) -> str:
    """Search the sports science knowledge base for evidence-based information
    about training load, injury prevention, recovery, and performance.
    Always use this first for any sports science question."""
    # This gets connected to the retriever in agent.py
    return f"Searching knowledge base for: {query}"

@tool
def calculate_acwr(acute_load: float, chronic_load: float) -> str:
    """Calculate the Acute Chronic Workload Ratio and assess injury risk level.
    Acute load is last 7 days training load in arbitrary units.
    Chronic load is last 28 days average training load in arbitrary units.
    Use this whenever the coach provides training load numbers."""
    if chronic_load == 0:
        return "Cannot calculate ACWR: chronic load is zero. Need at least 4 weeks of training data."

    acwr = round(acute_load / chronic_load, 2)

    if acwr < 0.8:
        risk = "LOW — undertraining risk"
        rec = "Gradually increase load by 10% per week."
    elif acwr <= 1.3:
        risk = "OPTIMAL — safe training zone"
        rec = "Maintain current load and monitor weekly."
    elif acwr <= 1.5:
        risk = "CAUTION — elevated injury risk"
        rec = "Reduce intensity. Prioritise recovery. Monitor HRV daily."
    else:
        risk = "HIGH DANGER — 2x injury risk"
        rec = "IMMEDIATE load reduction. Rest 48-72 hours minimum."

    return f"""ACWR Calculation:
Acute Load: {acute_load} AU
Chronic Load: {chronic_load} AU  
ACWR: {acwr}
Risk Level: {risk}
Recommendation: {rec}"""

@tool
def log_athlete_data(athlete_name: str, metric: str, value: str) -> str:
    """Log an athlete training metric or health measurement for tracking over time.
    Use whenever the coach mentions specific numbers like HRV, RPE,
    sleep hours, training load, heart rate, or any measurable data."""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "athlete": athlete_name,
        "metric": metric,
        "value": value
    }
    athlete_data_log.append(entry)
    return f"Logged: {athlete_name} | {metric} = {value} | {entry['timestamp']}"

@tool
def flag_professional_referral(athlete_name: str, reason: str, urgency: str) -> str:
    """Flag an athlete for professional medical or physiotherapy referral.
    Use IMMEDIATELY for chest pain, concussion, heat stroke, collapse,
    or any symptom requiring medical evaluation.
    Urgency must be: IMMEDIATE, URGENT, or ROUTINE."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    if urgency == "IMMEDIATE":
        return f"""🚨 EMERGENCY — {athlete_name}
Reason: {reason}
ACTION: Call emergency services (112 / 999 / 911) NOW.
Do not leave athlete unattended. Do not allow them to drive.
Timestamp: {timestamp}"""

    elif urgency == "URGENT":
        return f"""⚠️ URGENT REFERRAL — {athlete_name}
Reason: {reason}
ACTION: Sports medicine doctor within 24 hours.
Remove from all training until cleared.
Timestamp: {timestamp}"""

    else:
        return f"""📋 ROUTINE REFERRAL — {athlete_name}
Reason: {reason}
ACTION: Physiotherapy assessment within 1 week.
Timestamp: {timestamp}"""
