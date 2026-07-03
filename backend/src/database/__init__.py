"""Database exports."""
from src.database.repository import DatabaseError, PredictionRepository

PredictionDatabase = PredictionRepository

__all__ = ["DatabaseError", "PredictionRepository", "PredictionDatabase"]
