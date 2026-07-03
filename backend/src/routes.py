"""Compatibility route module."""
from src.api.routes import create_api_blueprint

create_routes = create_api_blueprint

__all__ = ["create_api_blueprint", "create_routes"]
