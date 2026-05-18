from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from app.core.exporter import save_model_run_artifacts
from app.core.modeling import run_machine_learning_model, run_statistical_model
from app.schemas.config import ModelingConfig
from app.storage.sqlite_store import append_task_step_log, complete_task_run, create_task_run, fail_task_run


def run_modeling_with_source(
    df: pd.DataFrame,
    config: ModelingConfig,
    *,
    source_name: str | None = None,
    dataset_id: str | None = None,
    artifact_root: str | Path | None = None,
    db_path: str | Path | None = None,
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
        db_path=db_path,
    )
    task_id = task_record["task_id"]

    try:
        append_task_step_log(
            task_id,
            step_name="准备特征矩阵",
            step_status="done",
            log_level="info",
            message=f"建模任务已创建，目标变量为“{config.target_column}”。",
            detail={
                "model_family": config.model_family,
                "algorithm": config.algorithm,
                "problem_type": config.problem_type,
            },
            db_path=db_path,
        )

        if config.model_family == "statistical":
            result = run_statistical_model(df, config)
        elif config.model_family == "machine_learning":
            result = run_machine_learning_model(df, config)
        else:
            raise ValueError("当前仅支持经典统计建模和机器学习建模。")

        append_task_step_log(
            task_id,
            step_name="模型训练",
            step_status="done",
            log_level="info",
            message=f"{config.algorithm} 训练完成。",
            detail={"metrics": result.get("metrics")},
            db_path=db_path,
        )

        if result.get("coefficient_table"):
            append_task_step_log(
                task_id,
                step_name="结果解释",
                step_status="done",
                log_level="info",
                message="统计建模结果解释已生成，包含系数、显著性与置信区间。",
                detail={"coefficient_count": len(result["coefficient_table"])},
                db_path=db_path,
            )
        elif result.get("feature_importance_table"):
            append_task_step_log(
                task_id,
                step_name="结果解释",
                step_status="done",
                log_level="info",
                message="机器学习特征重要性结果已生成。",
                detail={"feature_count": len(result["feature_importance_table"])},
                db_path=db_path,
            )

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
            append_task_step_log(
                task_id,
                step_name="结果归档",
                step_status="done",
                log_level="info",
                message="建模产物已保存到本地 artifacts 目录。",
                detail={"run_dir": artifacts.get("run_dir")},
                db_path=db_path,
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
            db_path=db_path,
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
        append_task_step_log(
            task_id,
            step_name="任务失败",
            step_status="failed",
            log_level="error",
            message=f"建模任务失败：{exc}",
            db_path=db_path,
        )
        fail_task_run(task_id, error_message=str(exc), db_path=db_path)
        raise


def _default_artifact_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "model_artifacts"
