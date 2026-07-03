"""SQLite persistence for prediction history."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.config import Config, FEATURE_NAMES
from src.logger import get_logger


class DatabaseError(RuntimeError):
    """Raised when SQLite operations fail."""


class PredictionDatabase:
    """Manage prediction history stored in SQLite."""

    def __init__(self, config: Config = Config()) -> None:
        self.config = config
        self.path: Path = config.database_path
        self.logger = get_logger(self.__class__.__name__, config)
        self.initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.path, timeout=10)
            connection.row_factory = sqlite3.Row
            yield connection
            connection.commit()
        except sqlite3.Error as exc:
            raise DatabaseError(str(exc)) from exc
        finally:
            if connection is not None:
                connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS PredictionHistory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mean REAL NOT NULL,
                    std REAL NOT NULL,
                    variance REAL NOT NULL,
                    min REAL NOT NULL,
                    max REAL NOT NULL,
                    energy REAL NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
        self.logger.info("SQLite database initialized at %s", self.path)

    def health_check(self) -> str:
        with self._connect() as connection:
            connection.execute("SELECT 1").fetchone()
        return "connected"

    def insert_prediction(self, features: dict[str, float], result: dict[str, Any]) -> int:
        timestamp = datetime.now(timezone.utc).isoformat()
        values = [features[name] for name in FEATURE_NAMES] + [
            result["prediction"],
            float(result["confidence"]),
            result["risk_level"],
            timestamp,
        ]
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO PredictionHistory (
                    mean, std, variance, min, max, energy, prediction, confidence, risk_level, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            return int(cursor.lastrowid)

    def get_history(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM PredictionHistory ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]

    def clear_history(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM PredictionHistory")
            return int(cursor.rowcount)
