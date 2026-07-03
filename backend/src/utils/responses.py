"""Consistent JSON response helpers."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from flask import jsonify


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def success_response(data: dict[str, Any], status_code: int = 200):
    payload = {"status": "success", "timestamp": utc_now_iso(), **data}
    return jsonify(payload), status_code


def error_response(error: str, message: str, status_code: int):
    return jsonify({
        "status": "error",
        "error": error,
        "message": message,
        "timestamp": utc_now_iso(),
        "http_status": status_code,
    }), status_code
