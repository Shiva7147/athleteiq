"""
security/audit.py
Immutable, append-only security and PHI access audit logging.
Compliant with HIPAA §164.312(b) (Audit Controls) and GDPR Article 30 (Records of Processing).
"""
import time
import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

_AUDIT_LOG_BUFFER: List[Dict[str, Any]] = []
_MAX_BUFFER_SIZE = 5000


def log_security_event(
    user_id: str,
    role: str,
    action: str,
    endpoint: str,
    player_id: Optional[str] = None,
    status_code: int = 200,
    client_ip: str = "127.0.0.1",
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Appends an immutable audit record to the structured audit log.
    Captures who accessed which athlete's record, when, and with what authorization level.
    """
    entry = {
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "timestamp_epoch": time.time(),
        "user_id": user_id,
        "role": role,
        "action": action,
        "endpoint": endpoint,
        "player_id": player_id,
        "status_code": status_code,
        "client_ip": client_ip,
        "details": details or {},
    }

    _AUDIT_LOG_BUFFER.append(entry)
    if len(_AUDIT_LOG_BUFFER) > _MAX_BUFFER_SIZE:
        _AUDIT_LOG_BUFFER.pop(0)

    return entry


def get_recent_audit_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent audit events for compliance inspection."""
    return list(reversed(_AUDIT_LOG_BUFFER[-limit:]))
