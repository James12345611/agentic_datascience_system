from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict
from uuid import uuid4

import pandas as pd

from app.schemas.config import PreprocessConfig


def save_run_artifacts(
    *,
    analysis: Dict,
    resolved_types: Dict[str, str],
    cleaned_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
    run_log: Dict[str, str],
    config: PreprocessConfig,
    artifact_root: str | Path,
    source_name: str | None = None,
) -> Dict:
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)

    source_stem = Path(source_name).stem if source_name else "manual"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{run_id}_{source_stem}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)

    cleaned_path = run_dir / "cleaned_data.csv"
    transformed_path = run_dir / "transformed_data.csv"
    config_path = run_dir / "preprocess_config.json"
    quality_path = run_dir / "quality_report.json"
    types_path = run_dir / "resolved_types.json"
    log_path = run_dir / "run_log.json"

    cleaned_df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
    transformed_df.to_csv(transformed_path, index=False, encoding="utf-8-sig")

    _write_json(config_path, config.model_dump())
    _write_json(quality_path, analysis["quality_report"])
    _write_json(types_path, resolved_types)
    _write_json(log_path, run_log)

    split_files: Dict[str, str] = {}
    for split_name, split_df in splits.items():
        split_path = run_dir / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False, encoding="utf-8-sig")
        split_files[split_name] = str(split_path)

    manifest = {
        "run_dir": str(run_dir),
        "files": {
            "cleaned_data": str(cleaned_path),
            "transformed_data": str(transformed_path),
            "preprocess_config": str(config_path),
            "quality_report": str(quality_path),
            "resolved_types": str(types_path),
            "run_log": str(log_path),
        },
        "split_files": split_files,
    }
    _write_json(run_dir / "manifest.json", manifest)
    return manifest


def save_model_run_artifacts(
    *,
    result_payload: Dict,
    config_payload: Dict,
    prediction_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artifact_root: str | Path,
    source_name: str | None = None,
) -> Dict:
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)

    source_stem = Path(source_name).stem if source_name else "manual"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{run_id}_{source_stem}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)

    config_path = run_dir / "modeling_config.json"
    result_path = run_dir / "modeling_result.json"
    prediction_path = run_dir / "prediction_result.csv"
    train_path = run_dir / "train_data.csv"
    test_path = run_dir / "test_data.csv"

    _write_json(config_path, config_payload)
    _write_json(result_path, result_payload)
    prediction_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    manifest = {
        "run_dir": str(run_dir),
        "files": {
            "modeling_config": str(config_path),
            "modeling_result": str(result_path),
            "prediction_result": str(prediction_path),
            "train_data": str(train_path),
            "test_data": str(test_path),
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    return manifest


def _write_json(path: Path, payload: Dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
