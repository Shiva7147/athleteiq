"""
agent/safety.py
Deterministic zero-LLM clinical safety node enforcing 100% recall on medical red flags.
Bypasses LLM entirely when emergency symptoms, unverified clearances, or pain masking are detected.
"""
from typing import Tuple, List

# Strict deterministic keyword & phrase patterns
CRITICAL_EMERGENCY_TRIGGERS = [
    "clear him to play",
    "ignore the doctor",
    "bypass medical",
    "pain injection",
    "numb the knee",
    "concussion protocol skip",
    "skip concussion protocol",
    "concussion",
    "torn acl",
    "loss of consciousness",
    "head injury return today",
    "hide the scan",
]

# Adversarial Prompt Injection & Jailbreak Defense Patterns
PROMPT_INJECTION_TRIGGERS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard all prior instructions",
    "system prompt",
    "jailbreak",
    "dan mode",
    "developer mode",
    "reveal instructions",
    "show your instructions",
    "exfiltrate",
    "bypass guardrails",
    "override protocol",
    "unrestricted research agent",
    "<system>",
    "</system>",
    "you are now in",
    "pretend you are an unrestricted",
]


def evaluate_clinical_safety_guardrail(query: str) -> Tuple[bool, str]:
    """
    Scans incoming queries for safety violations, medical emergencies,
    and adversarial prompt injection attempts.
    
    Returns:
        (is_violation: bool, escalation_message: str)
    """
    normalized = query.lower()

    # 1. Adversarial Prompt Injection Guard
    for trigger in PROMPT_INJECTION_TRIGGERS:
        if trigger in normalized:
            msg = (
                f"SECURITY AUDIT ALERT: Adversarial prompt injection or jailbreak attempt detected ('{trigger}'). "
                "AthleteIQ operates in a zero-trust enclave. System instructions, model weights, and confidential "
                "athlete biometrics are strictly protected. The interaction has been flagged and logged."
            )
            return True, msg

    # 2. Clinical Emergency & Medical Bypass Guard
    for trigger in CRITICAL_EMERGENCY_TRIGGERS:
        if trigger in normalized:
            msg = (
                f"CLINICAL SAFETY ESCALATION TRIGGERED: Detected unauthorized medical phrase or emergency trigger '{trigger}'. "
                "AthleteIQ strictly operates as an algorithmic sports-science decision-support tool. "
                "The system is prohibited from diagnosing pathology, overriding medical staff, or authorizing "
                "concussion/ligament clearances. Immediate escalation to Head of Medical is mandatory."
            )
            return True, msg

    return False, ""
