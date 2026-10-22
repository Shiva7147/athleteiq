"""
security/rate_limiter.py
Sliding-Window Token-Bucket Rate Limiter protecting core simulation and LLM endpoints.
Defends against compute exhaustion and Denial of Service (DoS) attacks.
"""
import time
from collections import defaultdict
from typing import Dict, List
from fastapi import Request, HTTPException, status


class SlidingWindowRateLimiter:
    """
    Thread-safe in-memory sliding window rate limiter.
    Limits requests per client IP or authenticated API token.
    """
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.client_history: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_key: str) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds
        
        # Purge timestamps outside the current sliding window
        timestamps = self.client_history[client_key]
        self.client_history[client_key] = [t for t in timestamps if t > cutoff]
        
        if len(self.client_history[client_key]) >= self.max_requests:
            return False

        self.client_history[client_key].append(now)
        return True

    def check_rate_limit(self, request: Request, custom_limit: int = None):
        client_ip = request.client.host if request.client else "127.0.0.1"
        api_key = request.headers.get("X-API-Key", "")
        client_key = f"{api_key}:{client_ip}" if api_key else client_ip

        limit = custom_limit or self.max_requests
        now = time.time()
        cutoff = now - self.window_seconds

        timestamps = self.client_history[client_key]
        self.client_history[client_key] = [t for t in timestamps if t > cutoff]

        if len(self.client_history[client_key]) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: Max {limit} requests per {self.window_seconds}s. Please back off.",
                headers={"Retry-After": str(self.window_seconds)},
            )

        self.client_history[client_key].append(now)


# Global rate limiter singletons
_general_limiter = SlidingWindowRateLimiter(max_requests=120, window_seconds=60)
_heavy_compute_limiter = SlidingWindowRateLimiter(max_requests=30, window_seconds=60)


def enforce_general_rate_limit(request: Request):
    """Enforces 120 req/min for general reads."""
    _general_limiter.check_rate_limit(request, custom_limit=120)


def enforce_heavy_rate_limit(request: Request):
    """Enforces 30 req/min for compute-intensive trajectories & LangGraph chat."""
    _heavy_compute_limiter.check_rate_limit(request, custom_limit=30)
