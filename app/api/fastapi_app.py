from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.schemas.config import ModelingConfig, PreprocessConfig, SplitConfig
from app.services.modeling_service import run_modeling_with_source
from app.services.platform_api_service import (
    build_dataset_catalog,
    build_dataset_dictionary,
    build_dataset_dictionary_revisions,
    build_dataset_preview,
    build_monitoring_bundle,
    build_overview_snapshot,
    build_task_center_snapshot,
    build_workbench_snapshot,
    save_dataset_dictionary_field,
)
from app.services.preprocess_service import analyze_dataframe, persist_raw_dataframe, run_preprocess_with_source
from app.storage.sqlite_store import (
    ensure_database,
    get_raw_dataset_entry,
    list_preprocessed_datasets,
    load_preprocessed_dataset,
    load_raw_dataset,
)


class DictionaryFieldPatchRequest(BaseModel):
    fieldName: str
    dataType: str | None = None
    semanticType: str | None = None
    unit: str | None = None
    softRange: str | None = None
    note: str | None = None
    summary: str | None = None
    actor: str = "前端人工微调"
    taskId: str | None = None


class PreprocessRunRequest(BaseModel):
    datasetId: str
    sourceName: str | None = None
    dropColumns: list[str] = Field(default_factory=list)
    missingNumeric: str = "median"
    missingCategorical: str = "mode"
    missingConstantValue: str | None = None
    outlierStrategy: str = "none"
    outlierLowerQuantile: float = 0.01
    outlierUpperQuantile: float = 0.99
    encoding: str = "onehot"
    scaling: str = "none"
    removeDuplicates: bool = True
    dropAllNullColumns: bool = True
    dropSingleValueColumns: bool = True
    saveArtifacts: bool = False
    enableSplit: bool = False
    splitTestSize: float = 0.2
    fieldMetadata: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ModelingRunRequest(BaseModel):
    datasetId: str
    preprocessRunId: str | None = None
    sourceName: str | None = None
    config: ModelingConfig


def create_app(*, db_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(
        title="统计平台 FastAPI",
        version="0.2.0",
        description="面向中文数据分析工作流的原型 API。",
    )
    app.state.db_path = str(db_path) if db_path else None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        resolved_path = ensure_database(_db_path(app))
        return {
            "status": "ok",
            "service": "statistic-platform-api",
            "dbPath": str(resolved_path),
        }

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        return build_overview_snapshot(db_path=_db_path(app))

    @app.get("/api/workbench")
    def workbench() -> dict[str, Any]:
        return build_workbench_snapshot(db_path=_db_path(app))

    @app.get("/api/datasets")
    def datasets() -> dict[str, Any]:
        return {
            "items": build_dataset_catalog(db_path=_db_path(app)),
        }

    @app.post("/api/datasets/upload")
    async def upload_dataset(
        file: UploadFile = File(...),
        dataset_name: str | None = Form(default=None),
    ) -> dict[str, Any]:
        db_path = _db_path(app)
        df = await _read_uploaded_dataframe(file)
        resolved_name = (dataset_name or "").strip() or Path(file.filename or "uploaded_dataset").stem

        saved = persist_raw_dataframe(
            df,
            dataset_name=resolved_name,
            source_file_name=file.filename,
            db_path=db_path,
        )
        analysis = analyze_dataframe(df)

        return {
            "datasetId": saved["dataset_id"],
            "datasetName": saved["dataset_name"],
            "sourceFileName": file.filename,
            "rowCount": saved["row_count"],
            "columnCount": saved["column_count"],
            "preview": build_dataset_preview(saved["dataset_id"], view="raw", limit=10, db_path=db_path),
            "inferredTypes": analysis["inferred_types"],
            "typeReviewSuggestions": analysis["type_review_suggestions"],
            "datasetSummary": analysis["quality_report"]["dataset_summary"],
        }

    @app.get("/api/datasets/{dataset_id}/preview")
    def dataset_preview(
        dataset_id: str,
        view: Literal["raw", "modeling"] = Query(default="raw"),
        limit: int = Query(default=10, ge=1, le=50),
    ) -> dict[str, Any]:
        try:
            return build_dataset_preview(dataset_id, view=view, limit=limit, db_path=_db_path(app))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/preprocess/run")
    def run_preprocess(payload: PreprocessRunRequest) -> dict[str, Any]:
        db_path = _db_path(app)
        dataset_entry = _get_dataset_or_404(payload.datasetId, db_path=db_path)
        df = load_raw_dataset(payload.datasetId, db_path=db_path)
        source_name = payload.sourceName or dataset_entry.get("source_file_name") or dataset_entry["dataset_name"]

        config = PreprocessConfig(
            drop_columns=payload.dropColumns,
            missing_numeric=payload.missingNumeric,
            missing_categorical=payload.missingCategorical,
            missing_constant_value=payload.missingConstantValue,
            outlier_strategy=payload.outlierStrategy,
            outlier_lower_quantile=payload.outlierLowerQuantile,
            outlier_upper_quantile=payload.outlierUpperQuantile,
            encoding=payload.encoding,
            scaling=payload.scaling,
            remove_duplicates=payload.removeDuplicates,
            drop_all_null_columns=payload.dropAllNullColumns,
            drop_single_value_columns=payload.dropSingleValueColumns,
            save_artifacts=payload.saveArtifacts,
            split=SplitConfig(test_size=payload.splitTestSize) if payload.enableSplit else None,
        )

        result = run_preprocess_with_source(
            df,
            config,
            source_name=source_name,
            dataset_id=payload.datasetId,
            persist_output=True,
            field_metadata=payload.fieldMetadata,
            db_path=db_path,
        )
        preprocess_storage = result["preprocess_storage"] or {}

        return {
            "datasetId": payload.datasetId,
            "datasetName": dataset_entry["dataset_name"],
            "taskId": result["task"]["task_id"],
            "taskStatus": result["task"]["status"],
            "preprocessRunId": preprocess_storage.get("preprocess_run_id"),
            "rawShape": list(df.shape),
            "cleanedShape": list(result["cleaned_df"].shape),
            "transformedShape": list(result["transformed_df"].shape),
            "validationIssueCount": sum(len(items) for items in result["validation_findings"].values()),
            "suggestionCount": len(result["analysis"]["type_review_suggestions"]),
            "preview": build_dataset_preview(payload.datasetId, view="modeling", limit=10, db_path=db_path),
        }

    @app.get("/api/tasks/summary")
    def task_summary() -> dict[str, Any]:
        return build_task_center_snapshot(db_path=_db_path(app))

    @app.get("/api/tasks")
    def tasks(
        task_type: str | None = Query(default=None),
        status: str | None = Query(default=None),
        dataset_id: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        snapshot = build_task_center_snapshot(db_path=_db_path(app))
        items = snapshot["tasks"]

        if task_type:
            items = [item for item in items if item["type"] == task_type]
        if status:
            items = [item for item in items if item["status"] == status]
        if dataset_id:
            items = [item for item in items if item.get("datasetId") == dataset_id]

        return {
            "items": items[:limit],
            "total": len(items),
        }

    @app.get("/api/tasks/{task_id}")
    def task_detail(task_id: str) -> dict[str, Any]:
        try:
            return build_monitoring_bundle(task_id, db_path=_db_path(app))["task"]
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/tasks/{task_id}/logs")
    def task_logs(task_id: str) -> dict[str, Any]:
        try:
            bundle = build_monitoring_bundle(task_id, db_path=_db_path(app))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "taskId": task_id,
            "logs": bundle["logs"],
        }

    @app.get("/api/tasks/{task_id}/monitoring")
    def task_monitoring(task_id: str) -> dict[str, Any]:
        try:
            return build_monitoring_bundle(task_id, db_path=_db_path(app))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/datasets/{dataset_id}/dictionary")
    def dataset_dictionary(dataset_id: str) -> dict[str, Any]:
        try:
            dataset_entry = get_raw_dataset_entry(dataset_id, db_path=_db_path(app))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return {
            "datasetId": dataset_id,
            "datasetName": dataset_entry["dataset_name"],
            "fields": build_dataset_dictionary(dataset_id, db_path=_db_path(app)),
            "revisions": build_dataset_dictionary_revisions(dataset_id, db_path=_db_path(app)),
        }

    @app.patch("/api/datasets/{dataset_id}/dictionary")
    def patch_dataset_dictionary(
        dataset_id: str,
        payload: DictionaryFieldPatchRequest,
    ) -> dict[str, Any]:
        try:
            dataset_entry = get_raw_dataset_entry(dataset_id, db_path=_db_path(app))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        result = save_dataset_dictionary_field(
            dataset_id,
            field_name=payload.fieldName,
            data_type=payload.dataType,
            semantic_type=payload.semanticType,
            unit=payload.unit,
            soft_range=payload.softRange,
            note=payload.note,
            summary=payload.summary,
            actor=payload.actor,
            task_id=payload.taskId,
            db_path=_db_path(app),
        )
        return {
            "datasetId": dataset_id,
            "datasetName": dataset_entry["dataset_name"],
            "fields": result["dictionary"],
            "revisions": result["revisions"],
            "revision": result["revision"],
        }

    @app.post("/api/modeling/run")
    def run_modeling(payload: ModelingRunRequest) -> dict[str, Any]:
        db_path = _db_path(app)
        dataset_entry = _get_dataset_or_404(payload.datasetId, db_path=db_path)
        dataset_frame, data_source = _load_modeling_frame(
            payload.datasetId,
            preprocess_run_id=payload.preprocessRunId,
            db_path=db_path,
        )

        result = run_modeling_with_source(
            dataset_frame,
            payload.config,
            source_name=payload.sourceName or dataset_entry["dataset_name"],
            dataset_id=payload.datasetId,
            db_path=db_path,
        )
        return {
            "dataSource": data_source,
            "task": result["task"],
            "result": result["result"],
            "artifacts": result["artifacts"],
        }

    return app


def _db_path(app: FastAPI) -> str | None:
    return app.state.db_path


def _get_dataset_or_404(dataset_id: str, *, db_path: str | None) -> dict[str, Any]:
    try:
        return get_raw_dataset_entry(dataset_id, db_path=db_path)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _load_modeling_frame(
    dataset_id: str,
    *,
    preprocess_run_id: str | None,
    db_path: str | None,
):
    if preprocess_run_id:
        try:
            return load_preprocessed_dataset(preprocess_run_id, db_path=db_path), "preprocessed_data"
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    for item in list_preprocessed_datasets(db_path=db_path, limit=500):
        if item["dataset_id"] == dataset_id:
            return (
                load_preprocessed_dataset(item["preprocess_run_id"], db_path=db_path),
                "preprocessed_data",
            )

    try:
        return load_raw_dataset(dataset_id, db_path=db_path), "raw_dataset"
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


async def _read_uploaded_dataframe(file: UploadFile) -> pd.DataFrame:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="上传文件为空。")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix == ".csv":
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                return pd.read_csv(BytesIO(payload), sep=None, engine="python", encoding=encoding)
            except Exception:
                continue
        raise HTTPException(status_code=400, detail="CSV 文件编码无法识别，请优先使用 UTF-8 或 GB18030。")

    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(BytesIO(payload))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Excel 文件读取失败：{exc}") from exc

    raise HTTPException(status_code=400, detail="当前仅支持 CSV、XLSX、XLS 文件。")


app = create_app()
