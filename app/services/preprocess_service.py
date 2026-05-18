from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from app.core.cleaner import clean_dataframe
from app.core.exporter import save_run_artifacts
from app.core.quality import build_quality_report
from app.core.splitter import split_dataframe
from app.core.transformer import transform_dataframe
from app.core.type_infer import infer_column_types
from app.core.validator import validate_columns
from app.schemas.config import PreprocessConfig
from app.storage.sqlite_store import (
    append_task_step_log,
    complete_task_run,
    create_task_run,
    fail_task_run,
    save_preprocessed_dataset,
    save_raw_dataset,
)


def analyze_dataframe(df: pd.DataFrame) -> Dict:
    inferred_types = infer_column_types(df)
    quality_report = build_quality_report(df, inferred_types)
    type_review_suggestions = build_type_review_suggestions(df, inferred_types)
    return {
        "inferred_types": inferred_types,
        "quality_report": quality_report,
        "type_review_suggestions": type_review_suggestions,
    }


def run_preprocess(df: pd.DataFrame, config: PreprocessConfig) -> Dict:
    return run_preprocess_with_source(df, config)


def run_preprocess_with_source(
    df: pd.DataFrame,
    config: PreprocessConfig,
    source_name: str | None = None,
    artifact_root: str | Path | None = None,
    dataset_id: str | None = None,
    persist_output: bool = False,
    field_metadata: Dict[str, dict] | None = None,
    db_path: str | Path | None = None,
) -> Dict:
    task_record = create_task_run(
        task_type="preprocess",
        dataset_id=dataset_id,
        source_name=source_name,
        request_payload={
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "save_artifacts": config.save_artifacts,
            "split_enabled": config.split is not None,
        },
        db_path=db_path,
    )
    task_id = task_record["task_id"]

    try:
        append_task_step_log(
            task_id,
            step_name="载入数据集",
            step_status="done",
            log_level="info",
            message=f"已载入数据集，共 {df.shape[0]} 行、{df.shape[1]} 列。",
            db_path=db_path,
        )

        analysis = analyze_dataframe(df)
        inferred_types = analysis["inferred_types"]
        suggestion_count = len(analysis["type_review_suggestions"])
        append_task_step_log(
            task_id,
            step_name="自动识别字段类型",
            step_status="done",
            log_level="warning" if suggestion_count > 0 else "info",
            message=(
                f"字段类型识别完成，检测到 {suggestion_count} 个建议人工确认的字段。"
                if suggestion_count > 0
                else "字段类型识别完成，当前未发现需要人工确认的字段。"
            ),
            detail={"suggestion_count": suggestion_count},
            db_path=db_path,
        )

        cleaned_df, clean_log, resolved_types = clean_dataframe(df, inferred_types, config)
        append_task_step_log(
            task_id,
            step_name="数据清洗",
            step_status="done",
            log_level="info",
            message=f"数据清洗完成，维度由 {list(df.shape)} 变为 {list(cleaned_df.shape)}。",
            detail={"clean_log": clean_log},
            db_path=db_path,
        )

        validation_findings = validate_columns(cleaned_df, resolved_types, config.column_configs)
        validation_issue_count = sum(len(items) for items in validation_findings.values())
        append_task_step_log(
            task_id,
            step_name="字段约束校验",
            step_status="done",
            log_level="warning" if validation_issue_count > 0 else "info",
            message=(
                f"字段约束校验完成，发现 {validation_issue_count} 条待关注问题。"
                if validation_issue_count > 0
                else "字段约束校验完成，当前未发现约束问题。"
            ),
            detail={"validation_findings": validation_findings},
            db_path=db_path,
        )

        transformed_df, transform_log = transform_dataframe(cleaned_df, resolved_types, config)
        append_task_step_log(
            task_id,
            step_name="特征转换",
            step_status="done",
            log_level="info",
            message=f"特征转换完成，维度由 {list(cleaned_df.shape)} 变为 {list(transformed_df.shape)}。",
            detail={"transform_log": transform_log},
            db_path=db_path,
        )

        splits = split_dataframe(transformed_df, config.split)
        split_keys = sorted(splits.keys())
        append_task_step_log(
            task_id,
            step_name="数据切分",
            step_status="done",
            log_level="info",
            message=f"数据切分完成，生成分片：{split_keys}。",
            detail={"split_keys": split_keys},
            db_path=db_path,
        )

        run_log = {**clean_log, **transform_log}
        artifacts = {}
        if config.save_artifacts:
            artifacts = save_run_artifacts(
                analysis=analysis,
                resolved_types=resolved_types,
                cleaned_df=cleaned_df,
                transformed_df=transformed_df,
                splits=splits,
                run_log=run_log,
                config=config,
                artifact_root=artifact_root or _default_artifact_root(),
                source_name=source_name,
            )
            append_task_step_log(
                task_id,
                step_name="本地产物保存",
                step_status="done",
                log_level="info",
                message="预处理产物已保存到本地 artifacts 目录。",
                detail={"run_dir": artifacts.get("run_dir")},
                db_path=db_path,
            )

        preprocess_storage = None
        if persist_output:
            if dataset_id is None:
                raise ValueError("persist_output=True 时必须提供 dataset_id。")
            preprocess_storage = save_preprocessed_dataset(
                transformed_df,
                dataset_id=dataset_id,
                task_id=task_id,
                resolved_types=resolved_types,
                field_metadata=field_metadata or {},
                preprocess_config=config.model_dump(),
                validation_findings=validation_findings,
                db_path=db_path,
            )
            append_task_step_log(
                task_id,
                step_name="结果落库",
                step_status="done",
                log_level="info",
                message="预处理结果已写入 SQLite。",
                detail=preprocess_storage,
                db_path=db_path,
            )

        task_summary = complete_task_run(
            task_id,
            result_payload={
                "status": "completed",
                "cleaned_shape": list(cleaned_df.shape),
                "transformed_shape": list(transformed_df.shape),
                "artifact_run_dir": artifacts.get("run_dir") if artifacts else None,
                "validation_issue_count": validation_issue_count,
                "preprocess_run_id": preprocess_storage["preprocess_run_id"] if preprocess_storage else None,
            },
            db_path=db_path,
        )

        return {
            "task": task_summary,
            "analysis": analysis,
            "resolved_types": resolved_types,
            "validation_findings": validation_findings,
            "cleaned_df": cleaned_df,
            "transformed_df": transformed_df,
            "splits": splits,
            "run_log": run_log,
            "artifacts": artifacts,
            "preprocess_storage": preprocess_storage,
        }
    except Exception as exc:
        append_task_step_log(
            task_id,
            step_name="任务失败",
            step_status="failed",
            log_level="error",
            message=f"预处理任务失败：{exc}",
            db_path=db_path,
        )
        fail_task_run(task_id, error_message=str(exc), db_path=db_path)
        raise


def persist_raw_dataframe(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    source_file_name: str | None = None,
    db_path: str | Path | None = None,
) -> Dict:
    return save_raw_dataset(
        df,
        dataset_name=dataset_name,
        source_file_name=source_file_name,
        db_path=db_path,
    )


def persist_preprocessed_dataframe(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    task_id: str | None = None,
    resolved_types: Dict[str, str],
    field_metadata: Dict[str, dict],
    preprocess_config: Dict,
    validation_findings: Dict[str, list[str]],
    db_path: str | Path | None = None,
) -> Dict:
    return save_preprocessed_dataset(
        df,
        dataset_id=dataset_id,
        task_id=task_id,
        resolved_types=resolved_types,
        field_metadata=field_metadata,
        preprocess_config=preprocess_config,
        validation_findings=validation_findings,
        db_path=db_path,
    )


def _default_artifact_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "artifacts"


def build_type_review_suggestions(df: pd.DataFrame, inferred_types: Dict[str, str]) -> Dict[str, list[str]]:
    suggestions: Dict[str, list[str]] = {}
    row_count = max(len(df), 1)

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        column_suggestions: list[str] = []
        inferred_type = inferred_types.get(column, "categorical")
        unique_count = int(non_null.nunique(dropna=True))
        unique_ratio = unique_count / row_count
        missing_ratio = float(series.isna().mean())

        if missing_ratio > 0.3:
            column_suggestions.append("缺失比例较高，建议人工确认该字段是否保留以及采用何种填补方式。")
        if inferred_type == "id_like":
            column_suggestions.append("该字段疑似唯一标识列，通常不建议直接用于建模。")
        if inferred_type == "numerical" and 2 <= unique_count <= 10:
            column_suggestions.append("该字段被识别为数值型，但唯一值较少，也可能更适合作为分类型变量。")
        if inferred_type == "categorical" and unique_ratio > 0.8:
            column_suggestions.append("该字段被识别为分类型，但唯一值比例较高，可能更接近文本列或标识列。")
        if inferred_type == "datetime":
            column_suggestions.append("建议确认该字段的时间粒度，并判断是否需要衍生年、月、日等时间特征。")
        if inferred_type == "text":
            column_suggestions.append("文本字段当前仅做基础保留，建议确认是否需要后续文本清洗或删除。")

        if column_suggestions:
            suggestions[column] = column_suggestions

    return suggestions
