"""Middleware exports."""
from src.middleware.security import register_security_headers

__all__ = ["register_security_headers"]
