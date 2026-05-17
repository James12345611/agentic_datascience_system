from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "statistic_platform.db"


def ensure_database(db_path: str | Path | None = None) -> Path:
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_dataset (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL UNIQUE,
                dataset_name TEXT NOT NULL,
                source_file_name TEXT,
                upload_time TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                column_count INTEGER NOT NULL,
                schema_json TEXT NOT NULL,
                data_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS preprocessed_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                preprocess_run_id TEXT NOT NULL UNIQUE,
                dataset_id TEXT NOT NULL,
                task_id TEXT,
                preprocess_time TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                column_count INTEGER NOT NULL,
                resolved_types_json TEXT NOT NULL,
                field_metadata_json TEXT NOT NULL,
                preprocess_config_json TEXT NOT NULL,
                validation_findings_json TEXT NOT NULL,
                data_json TEXT NOT NULL
            )
            """
        )
        _ensure_column(conn, "preprocessed_data", "task_id", "TEXT")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_run (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL UNIQUE,
                task_type TEXT NOT NULL,
                dataset_id TEXT,
                status TEXT NOT NULL,
                created_time TEXT NOT NULL,
                updated_time TEXT NOT NULL,
                source_name TEXT,
                request_payload_json TEXT,
                result_payload_json TEXT,
                error_message TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_preprocessed_dataset_id ON preprocessed_data(dataset_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_preprocessed_task_id ON preprocessed_data(task_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_task_run_dataset_id ON task_run(dataset_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_task_run_type_status ON task_run(task_type, status)"
        )
        conn.commit()

    return path


def save_raw_dataset(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    source_file_name: str | None = None,
    db_path: str | Path | None = None,
) -> Dict[str, Any]:
    path = ensure_database(db_path)
    dataset_id = uuid4().hex
    upload_time = _now_iso()
    schema_json = json.dumps(_build_schema(df), ensure_ascii=False)
    data_json = _dataframe_to_json(df)

    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO raw_dataset (
                dataset_id, dataset_name, source_file_name, upload_time,
                row_count, column_count, schema_json, data_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                dataset_name,
                source_file_name,
                upload_time,
                int(df.shape[0]),
                int(df.shape[1]),
                schema_json,
                data_json,
            ),
        )
        conn.commit()

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "db_path": str(path),
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
    }


def save_preprocessed_dataset(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    task_id: str | None = None,
    resolved_types: Dict[str, str],
    field_metadata: Dict[str, Any],
    preprocess_config: Dict[str, Any],
    validation_findings: Dict[str, Any],
    db_path: str | Path | None = None,
) -> Dict[str, Any]:
    path = ensure_database(db_path)
    preprocess_run_id = uuid4().hex
    preprocess_time = _now_iso()

    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO preprocessed_data (
                preprocess_run_id, dataset_id, task_id, preprocess_time, row_count, column_count,
                resolved_types_json, field_metadata_json, preprocess_config_json,
                validation_findings_json, data_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                preprocess_run_id,
                dataset_id,
                task_id,
                preprocess_time,
                int(df.shape[0]),
                int(df.shape[1]),
                json.dumps(resolved_types, ensure_ascii=False),
                json.dumps(field_metadata, ensure_ascii=False, default=str),
                json.dumps(preprocess_config, ensure_ascii=False, default=str),
                json.dumps(validation_findings, ensure_ascii=False, default=str),
                _dataframe_to_json(df),
            ),
        )
        conn.commit()

    return {
        "preprocess_run_id": preprocess_run_id,
        "dataset_id": dataset_id,
        "task_id": task_id,
        "db_path": str(path),
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
    }


def create_task_run(
    *,
    task_type: str,
    dataset_id: str | None = None,
    source_name: str | None = None,
    request_payload: Dict[str, Any] | None = None,
    db_path: str | Path | None = None,
) -> Dict[str, Any]:
    path = ensure_database(db_path)
    task_id = uuid4().hex
    created_time = _now_iso()
    request_payload_json = json.dumps(request_payload or {}, ensure_ascii=False, default=str)

    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO task_run (
                task_id, task_type, dataset_id, status, created_time, updated_time,
                source_name, request_payload_json, result_payload_json, error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                task_type,
                dataset_id,
                "running",
                created_time,
                created_time,
                source_name,
                request_payload_json,
                None,
                None,
            ),
        )
        conn.commit()

    return {
        "task_id": task_id,
        "task_type": task_type,
        "dataset_id": dataset_id,
        "status": "running",
        "created_time": created_time,
        "updated_time": created_time,
        "source_name": source_name,
        "db_path": str(path),
    }


def complete_task_run(
    task_id: str,
    *,
    result_payload: Dict[str, Any],
    db_path: str | Path | None = None,
) -> Dict[str, Any]:
    path = ensure_database(db_path)
    updated_time = _now_iso()
    result_payload_json = json.dumps(result_payload, ensure_ascii=False, default=str)

    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            UPDATE task_run
            SET status = ?, updated_time = ?, result_payload_json = ?, error_message = ?
            WHERE task_id = ?
            """,
            ("completed", updated_time, result_payload_json, None, task_id),
        )
        conn.commit()

    return get_task_run(task_id, db_path=path)


def fail_task_run(
    task_id: str,
    *,
    error_message: str,
    db_path: str | Path | None = None,
) -> Dict[str, Any]:
    path = ensure_database(db_path)
    updated_time = _now_iso()

    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            UPDATE task_run
            SET status = ?, updated_time = ?, error_message = ?
            WHERE task_id = ?
            """,
            ("failed", updated_time, error_message, task_id),
        )
        conn.commit()

    return get_task_run(task_id, db_path=path)


def get_task_run(task_id: str, db_path: str | Path | None = None) -> Dict[str, Any]:
    path = ensure_database(db_path)
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            """
            SELECT task_id, task_type, dataset_id, status, created_time, updated_time,
                   source_name, request_payload_json, result_payload_json, error_message
            FROM task_run
            WHERE task_id = ?
            """,
            (task_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"task_run not found: {task_id}")

    return {
        "task_id": row[0],
        "task_type": row[1],
        "dataset_id": row[2],
        "status": row[3],
        "created_time": row[4],
        "updated_time": row[5],
        "source_name": row[6],
        "request_payload_json": json.loads(row[7]) if row[7] else {},
        "result_payload_json": json.loads(row[8]) if row[8] else None,
        "error_message": row[9],
    }


def list_task_runs(
    db_path: str | Path | None = None,
    *,
    task_type: str | None = None,
    dataset_id: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[Dict[str, Any]]:
    path = ensure_database(db_path)
    query = """
        SELECT task_id, task_type, dataset_id, status, created_time, updated_time,
               source_name, request_payload_json, result_payload_json, error_message
        FROM task_run
        WHERE 1 = 1
    """
    params: list[Any] = []

    if task_type:
        query += " AND task_type = ?"
        params.append(task_type)
    if dataset_id:
        query += " AND dataset_id = ?"
        params.append(dataset_id)
    if status:
        query += " AND status = ?"
        params.append(status)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(path) as conn:
        rows = conn.execute(query, params).fetchall()

    return [
        {
            "task_id": row[0],
            "task_type": row[1],
            "dataset_id": row[2],
            "status": row[3],
            "created_time": row[4],
            "updated_time": row[5],
            "source_name": row[6],
            "request_payload_json": json.loads(row[7]) if row[7] else {},
            "result_payload_json": json.loads(row[8]) if row[8] else None,
            "error_message": row[9],
        }
        for row in rows
    ]


def load_raw_dataset(dataset_id: str, db_path: str | Path | None = None) -> pd.DataFrame:
    path = ensure_database(db_path)
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT data_json FROM raw_dataset WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"raw_dataset not found: {dataset_id}")
    return _json_to_dataframe(row[0])


def list_raw_datasets(db_path: str | Path | None = None, limit: int = 100) -> list[Dict[str, Any]]:
    path = ensure_database(db_path)
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            """
            SELECT dataset_id, dataset_name, source_file_name, upload_time, row_count, column_count
            FROM raw_dataset
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        {
            "dataset_id": row[0],
            "dataset_name": row[1],
            "source_file_name": row[2],
            "upload_time": row[3],
            "row_count": row[4],
            "column_count": row[5],
        }
        for row in rows
    ]


def list_preprocessed_datasets(
    db_path: str | Path | None = None, limit: int = 100
) -> list[Dict[str, Any]]:
    path = ensure_database(db_path)
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            """
            SELECT preprocess_run_id, dataset_id, task_id, preprocess_time, row_count, column_count
            FROM preprocessed_data
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        {
            "preprocess_run_id": row[0],
            "dataset_id": row[1],
            "task_id": row[2],
            "preprocess_time": row[3],
            "row_count": row[4],
            "column_count": row[5],
        }
        for row in rows
    ]


def get_raw_dataset_entry(dataset_id: str, db_path: str | Path | None = None) -> Dict[str, Any]:
    path = ensure_database(db_path)
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            """
            SELECT dataset_id, dataset_name, source_file_name, upload_time, row_count, column_count, schema_json
            FROM raw_dataset
            WHERE dataset_id = ?
            """,
            (dataset_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"raw_dataset not found: {dataset_id}")

    return {
        "dataset_id": row[0],
        "dataset_name": row[1],
        "source_file_name": row[2],
        "upload_time": row[3],
        "row_count": row[4],
        "column_count": row[5],
        "schema_json": json.loads(row[6]),
    }


def load_preprocessed_dataset(
    preprocess_run_id: str,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    path = ensure_database(db_path)
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT data_json FROM preprocessed_data WHERE preprocess_run_id = ?",
            (preprocess_run_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"preprocessed_data not found: {preprocess_run_id}")
    return _json_to_dataframe(row[0])


def get_preprocessed_entry(
    preprocess_run_id: str,
    db_path: str | Path | None = None,
) -> Dict[str, Any]:
    path = ensure_database(db_path)
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            """
            SELECT preprocess_run_id, dataset_id, task_id, preprocess_time, row_count, column_count,
                   resolved_types_json, field_metadata_json, preprocess_config_json, validation_findings_json
            FROM preprocessed_data
            WHERE preprocess_run_id = ?
            """,
            (preprocess_run_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"preprocessed_data not found: {preprocess_run_id}")

    return {
        "preprocess_run_id": row[0],
        "dataset_id": row[1],
        "task_id": row[2],
        "preprocess_time": row[3],
        "row_count": row[4],
        "column_count": row[5],
        "resolved_types_json": json.loads(row[6]),
        "field_metadata_json": json.loads(row[7]),
        "preprocess_config_json": json.loads(row[8]),
        "validation_findings_json": json.loads(row[9]),
    }


def find_preprocessed_entry_by_task_id(
    task_id: str,
    db_path: str | Path | None = None,
) -> Dict[str, Any] | None:
    path = ensure_database(db_path)
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            """
            SELECT preprocess_run_id, dataset_id, task_id, preprocess_time, row_count, column_count,
                   resolved_types_json, field_metadata_json, preprocess_config_json, validation_findings_json
            FROM preprocessed_data
            WHERE task_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (task_id,),
        ).fetchone()
    if row is None:
        return None

    return {
        "preprocess_run_id": row[0],
        "dataset_id": row[1],
        "task_id": row[2],
        "preprocess_time": row[3],
        "row_count": row[4],
        "column_count": row[5],
        "resolved_types_json": json.loads(row[6]),
        "field_metadata_json": json.loads(row[7]),
        "preprocess_config_json": json.loads(row[8]),
        "validation_findings_json": json.loads(row[9]),
    }


def _build_schema(df: pd.DataFrame) -> Dict[str, str]:
    return {column: str(dtype) for column, dtype in df.dtypes.items()}


def _dataframe_to_json(df: pd.DataFrame) -> str:
    return df.to_json(orient="records", force_ascii=False, date_format="iso")


def _json_to_dataframe(payload: str) -> pd.DataFrame:
    records = json.loads(payload)
    return pd.DataFrame(records)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, definition: str) -> None:
    existing_columns = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in existing_columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")
