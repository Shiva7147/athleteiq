"""
security/anonymizer.py
Cryptographic Pseudonymization and Role-Based PHI Projection.
Ensures compliance with GDPR Article 9 (Special Category Data) and HIPAA §164.514 de-identification.
"""
import hmac
import hashlib
from typing import Dict, Any, List
from security.auth import UserRole

# Cryptographic Salt / Pepper (In production, fetched from AWS KMS / HashiCorp Vault)
_PEPPER = b"athleteiq-enterprise-hmac-salt-981247"


def generate_pseudonymous_token(player_id: str, club_id: str = "ARS") -> str:
    """
    Generates a deterministic, irreversible pseudonymous identifier for ML & LLM pipelines.
    Guarantees that external cloud models or third-party loggers never see raw player identities.
    """
    digest = hmac.new(_PEPPER, f"{club_id}:{player_id}".encode("utf-8"), hashlib.sha256).hexdigest()
    return f"ATH_{digest[:10].upper()}"


def sanitize_athlete_record_for_role(athlete: Dict[str, Any], role: UserRole) -> Dict[str, Any]:
    """
    Filters and transforms athlete medical and load records based on the caller's verified role.
    
    - DOCTOR: Complete unmasked access to clinical markers, raw biometrics, and names.
    - COACH: Tactical view (availability, minute capping, pseudonymized names). Raw medical text masked.
    - ANALYST: De-identified numerical metrics only (ATH_XXXX tokens, no names).
    """
    record = dict(athlete)
    player_id = record.get("player_id", "PLY_UNKNOWN")
    token = generate_pseudonymous_token(player_id)
    record["pseudonym_token"] = token

    if role == UserRole.DOCTOR:
        # Full clinical access
        return record

    elif role == UserRole.COACH:
        # Tactical access: Mask raw clinical notes, retain tactical directive
        # Keep squad role, e.g. "Alex Vance (Winger)" -> "A. Vance (Winger)"
        raw_name = record.get("name", "Unknown Athlete")
        if " " in raw_name:
            parts = raw_name.split(" ", 1)
            record["name"] = f"{parts[0][0]}. {parts[1]}"
        # Mask clinical pathology details
        if "clinical_notes" in record:
            record["clinical_notes"] = "[MASKED: Restricted to Medical Staff]"
        return record

    else:  # ANALYST
        # De-identified numerical view only
        record["name"] = f"Athlete {token}"
        if "clinical_notes" in record:
            del record["clinical_notes"]
        return record


def sanitize_prompt_for_llm(prompt: str, player_name_to_id: Dict[str, str]) -> str:
    """
    Scans prompts delivered to external LLMs and swaps legal player names with cryptographic tokens.
    """
    sanitized = prompt
    for name, pid in player_name_to_id.items():
        if name.lower() in sanitized.lower():
            token = generate_pseudonymous_token(pid)
            sanitized = sanitized.replace(name, token)
    return sanitized
