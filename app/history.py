"""Local, opt-in prediction history for the FedFairChain research prototype.

This module stores prediction visits in a separate SQLite database.  History is
never used for training or automatic future-risk prediction.  It exists only so
a user can revisit prior inputs and compare model outputs over time.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "app" / "fedfairchain_history.db"
PATIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def get_db_path() -> Path:
    override = os.getenv("FEDFAIRCHAIN_HISTORY_DB")
    return Path(override) if override else DEFAULT_DB_PATH


def validate_patient_id(patient_id: str) -> str:
    value = (patient_id or "").strip()
    if not PATIENT_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "Anonymous Patient ID must be 1-32 characters and use only "
            "letters, numbers, hyphens or underscores."
        )
    return value


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_history_db(db_path: Path | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                inputs_json TEXT NOT NULL,
                known_diagnoses_json TEXT NOT NULL,
                results_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prediction_history_patient "
            "ON prediction_history(patient_id, id DESC)"
        )


def _compact_results(prediction: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for target in prediction.get("chain_order", []):
        result = prediction["results"][target]
        compact[target] = {
            "source": result.get("source"),
            "known_status": result.get("known_status"),
            "probability_percent": result.get("probability_percent"),
            "threshold_percent": result.get("threshold_percent"),
            "primary_interpretation": result.get("primary_interpretation"),
            "prediction": result.get("prediction"),
        }
    return compact


def save_prediction(
    patient_id: str,
    patient: dict[str, Any],
    known_diagnoses: dict[str, Any],
    prediction: dict[str, Any],
    db_path: Path | None = None,
) -> int:
    """Save one server-generated prediction visit and return its record id."""
    patient_id = validate_patient_id(patient_id)
    initialize_history_db(db_path)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO prediction_history (
                patient_id, created_at, inputs_json,
                known_diagnoses_json, results_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                created_at,
                json.dumps(patient, allow_nan=False),
                json.dumps(known_diagnoses, allow_nan=False),
                json.dumps(_compact_results(prediction), allow_nan=False),
            ),
        )
        return int(cursor.lastrowid)


def get_patient_history(
    patient_id: str,
    limit: int = 20,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    patient_id = validate_patient_id(patient_id)
    initialize_history_db(db_path)
    limit = max(1, min(int(limit), 50))
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, patient_id, created_at, inputs_json,
                   known_diagnoses_json, results_json
            FROM prediction_history
            WHERE patient_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (patient_id, limit),
        ).fetchall()
    return [
        {
            "record_id": int(row["id"]),
            "patient_id": row["patient_id"],
            "created_at": row["created_at"],
            "inputs": json.loads(row["inputs_json"]),
            "known_diagnoses": json.loads(row["known_diagnoses_json"]),
            "results": json.loads(row["results_json"]),
        }
        for row in rows
    ]


def delete_patient_history(
    patient_id: str,
    db_path: Path | None = None,
) -> int:
    patient_id = validate_patient_id(patient_id)
    initialize_history_db(db_path)
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM prediction_history WHERE patient_id = ?",
            (patient_id,),
        )
        return int(cursor.rowcount)
