"""SQLite repository for prediction history."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.config import Config
from src.logging import get_logger


class DatabaseError(RuntimeError):
    """Raised when SQLite operations fail."""


class PredictionRepository:
    """Repository pattern wrapper around SQLite prediction history."""

    def __init__(self, config: Config = Config()) -> None:
        self.config = config
        self.path: Path = config.database_path
        self.logger = get_logger(self.__class__.__name__, config)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
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
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""
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
                    probability_normal REAL,
                    probability_abnormal REAL,
                    processing_time_ms REAL,
                    client_ip TEXT,
                    api_version TEXT,
                    model_version TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            self._migrate_columns(connection)
            connection.execute("CREATE INDEX IF NOT EXISTS idx_prediction_timestamp ON PredictionHistory(timestamp)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_prediction_result ON PredictionHistory(prediction)")
        self.logger.info("database_initialized")

    def _migrate_columns(self, connection: sqlite3.Connection) -> None:
        existing = {row[1] for row in connection.execute("PRAGMA table_info(PredictionHistory)").fetchall()}
        migrations = {
            "probability_normal": "REAL",
            "probability_abnormal": "REAL",
            "processing_time_ms": "REAL",
            "client_ip": "TEXT",
            "api_version": "TEXT",
            "model_version": "TEXT",
        }
        for column, column_type in migrations.items():
            if column not in existing:
                connection.execute(f"ALTER TABLE PredictionHistory ADD COLUMN {column} {column_type}")

    def health_check(self) -> str:
        with self.connect() as connection:
            connection.execute("SELECT 1").fetchone()
        return "connected"

    def save_prediction(self, features: dict[str, float], result: dict[str, Any], client_ip: str | None, api_version: str) -> int:
        probability = result.get("probability") or result.get("prediction_probability") or {}
        values = (
            features["mean"], features["std"], features["variance"], features["min"], features["max"], features["energy"],
            result["prediction"], float(result["confidence"]), result["risk_level"],
            float(probability.get("Normal", 0.0)), float(probability.get("Abnormal", 0.0)),
            float(result.get("processing_time_ms", 0.0)), client_ip, api_version, result.get("model_version"),
            result.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        )
        with self.connect() as connection:
            cursor = connection.execute("""
                INSERT INTO PredictionHistory (
                    mean, std, variance, min, max, energy, prediction, confidence, risk_level,
                    probability_normal, probability_abnormal, processing_time_ms, client_ip, api_version, model_version, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, values)
            return int(cursor.lastrowid)

    def get_history(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM PredictionHistory ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]

    def clear_history(self) -> int:
        with self.connect() as connection:
            cursor = connection.execute("DELETE FROM PredictionHistory")
            return int(cursor.rowcount)
