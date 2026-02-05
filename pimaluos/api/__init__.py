"""
PIMALUOS API Module

Contains FastAPI backend server and WebSocket handlers.
"""

from pimaluos.api.server import app, create_app

__all__ = [
    "app",
    "create_app",
]
