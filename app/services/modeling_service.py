from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from app.core.exporter import save_model_run_artifacts
from app.core.modeling import run_machine_learning_model, run_statistical_model
from app.schemas.config import ModelingConfig
from app.storage.sqlite_store import complete_task_run, create_task_run, fail_task_run


def run_modeling_with_source(
    df: pd.DataFrame,
    config: ModelingConfig,
    *,
    source_name: str | None = None,
    dataset_id: str | None = None,
    artifact_root: str | Path | None = None,
) -> Dict[str, Any]:
    task_record = create_task_run(
        task_type="modeling",
        dataset_id=dataset_id,
        source_name=source_name,
        request_payload={
            "model_family": config.model_family,
            "algorithm": config.algorithm,
            "problem_type": config.problem_type,
            "target_column": config.target_column,
            "feature_count": len(config.feature_columns),
        },
    )
    task_id = task_record["task_id"]

    try:
        if config.model_family == "statistical":
            result = run_statistical_model(df, config)
        elif config.model_family == "machine_learning":
            result = run_machine_learning_model(df, config)
        else:
            raise ValueError("当前仅支持经典统计建模和机器学习建模。")

        artifacts = {}
        raw_artifacts = result.pop("artifacts")
        if config.save_artifacts:
            artifacts = save_model_run_artifacts(
                result_payload={key: value for key, value in result.items() if key != "summary_text"},
                config_payload=config.model_dump(),
                prediction_df=raw_artifacts.prediction_frame,
                train_df=raw_artifacts.train_frame,
                test_df=raw_artifacts.test_frame,
                artifact_root=artifact_root or _default_artifact_root(),
                source_name=source_name,
            )

        task_summary = complete_task_run(
            task_id,
            result_payload={
                "status": "completed",
                "model_family": config.model_family,
                "algorithm": config.algorithm,
                "problem_type": config.problem_type,
                "target_column": config.target_column,
                "sample_size": result.get("sample_size"),
                "metrics": result.get("metrics"),
                "artifact_run_dir": artifacts.get("run_dir") if artifacts else None,
            },
        )

        return {
            "task": task_summary,
            "result": result,
            "prediction_df": raw_artifacts.prediction_frame,
            "train_df": raw_artifacts.train_frame,
            "test_df": raw_artifacts.test_frame,
            "artifacts": artifacts,
        }
    except Exception as exc:
        fail_task_run(task_id, error_message=str(exc))
        raise


def _default_artifact_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "model_artifacts"
