"""
api/index.py
Vercel Serverless Function Entry Point for AthleteIQ 2.0.
Exports the FastAPI application instance for Vercel Python runtime.
"""
import sys
import os

# Ensure the project root directory is on the Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api.routes import app

# Vercel looks for 'app' as the ASGI application entry point
__all__ = ["app"]
