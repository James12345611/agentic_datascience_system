from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any, Dict

from app.core.type_infer import infer_column_types
from app.storage.sqlite_store import (
    append_task_step_log,
    create_dataset_dictionary_revision,
    find_preprocessed_entry_by_task_id,
    get_preprocessed_entry,
    get_raw_dataset_entry,
    get_task_run,
    list_dataset_dictionary_overrides,
    list_dataset_dictionary_revisions,
    list_preprocessed_datasets,
    list_raw_datasets,
    list_task_runs,
    list_task_step_logs,
    load_preprocessed_dataset,
    load_raw_dataset,
    upsert_dataset_dictionary_override,
)


TYPE_LABELS = {
    "numerical": "数值",
    "categorical": "分类",
    "datetime": "时间",
    "boolean": "布尔",
    "text": "文本",
    "id_like": "标识",
}

SEMANTIC_LABELS = {
    "general": "普通字段",
    "proportion": "比例",
    "amount": "金额",
    "count": "计数",
    "score": "评分",
    "index": "指数",
    "duration": "时长",
    "nominal_category": "普通分类",
    "ordinal_category": "有序分类",
    "geographic_region": "地理区域",
    "industry_category": "行业分类",
    "administrative_region": "行政区域",
    "text_label": "文本标签",
    "custom": "自定义",
}

TASK_TYPE_LABELS = {
    "preprocess": "预处理",
    "modeling": "建模",
    "diagnostic": "统计诊断",
}

LOG_LEVEL_LABELS = {
    "info": "信息",
    "warning": "警告",
    "error": "错误",
}


def build_overview_snapshot(db_path: str | Path | None = None) -> Dict[str, Any]:
    tasks = list_task_runs(db_path=db_path, limit=300)
    raw_items = list_raw_datasets(db_path=db_path, limit=300)
    processed_items = list_preprocessed_datasets(db_path=db_path, limit=300)
    task_items = [_build_task_item(task, raw_items, db_path=db_path) for task in tasks]

    success_count = sum(1 for task in tasks if task["status"] == "completed")
    success_rate = (success_count / len(tasks) * 100) if tasks else 0.0
    field_revision_count = sum(len(list_dataset_dictionary_revisions(item["dataset_id"], db_path=db_path)) for item in raw_items[:50])
    recent_task_count = sum(
        1 for task in tasks if _parse_time(task["created_time"]) >= datetime.now() - timedelta(days=1)
    )

    return {
        "headline": "把预处理、诊断、建模和人工修正放进同一条中文数据工作流",
        "subtitle": "当前页面已经改为真实 FastAPI 数据源；如果数据库暂无记录，页面会显示空态或低占位信息。",
        "heroStats": [
            {"label": "原始数据集", "value": str(len(raw_items))},
            {"label": "预处理版本", "value": str(len(processed_items))},
            {"label": "任务快照数", "value": str(len(tasks))},
        ],
        "summaryCards": [
            {
                "title": "任务总量",
                "value": str(len(tasks)),
                "delta": f"近 24 小时 {recent_task_count} 个",
                "note": "统一包含预处理、统计诊断和建模任务。",
                "tone": "teal",
            },
            {
                "title": "运行成功率",
                "value": f"{success_rate:.1f}%",
                "delta": f"成功 {success_count} / 总计 {len(tasks)}",
                "note": "后续可拆成模块级成功率与耗时统计。",
                "tone": "amber",
            },
            {
                "title": "字段人工修正数",
                "value": str(field_revision_count),
                "delta": "来自字段字典修订历史",
                "note": "这部分会逐步成为 agent 学习与规则优化的依据。",
                "tone": "copper",
            },
            {
                "title": "预处理版本数",
                "value": str(len(processed_items)),
                "delta": f"原始数据集 {len(raw_items)} 个",
                "note": "后续可扩展为数据版本链与回滚能力。",
                "tone": "slate",
            },
        ],
        "trend": _build_recent_task_trend(tasks),
        "ranking": _build_model_ranking(tasks),
        "alerts": _build_recent_alerts(tasks, raw_items, db_path=db_path),
        "tasks": task_items[:3],
    }


def build_task_center_snapshot(db_path: str | Path | None = None) -> Dict[str, Any]:
    tasks = list_task_runs(db_path=db_path, limit=300)
    raw_items = list_raw_datasets(db_path=db_path, limit=300)
    task_items = [_build_task_item(task, raw_items, db_path=db_path) for task in tasks]

    stage_counter = Counter(item["stage"] for item in task_items if item["stage"])
    status_counter = Counter(item["status"] for item in task_items)

    return {
        "tasks": task_items,
        "stageStats": [{"label": label, "value": value} for label, value in stage_counter.most_common(8)],
        "statusStats": [
            {"label": _status_label(status), "value": value}
            for status, value in status_counter.items()
        ],
    }


def build_monitoring_bundle(task_id: str, db_path: str | Path | None = None) -> Dict[str, Any]:
    task = get_task_run(task_id, db_path=db_path)
    raw_items = list_raw_datasets(db_path=db_path, limit=300)
    task_item = _build_task_item(task, raw_items, db_path=db_path)
    logs = list_task_step_logs(task_id, db_path=db_path)
    if not logs:
        logs = _synthesize_task_logs(task)

    dataset_id = task.get("dataset_id")
    tuning_fields = build_dataset_dictionary(dataset_id, db_path=db_path) if dataset_id else []
    revisions = build_dataset_dictionary_revisions(dataset_id, db_path=db_path) if dataset_id else []

    return {
        "task": task_item,
        "timeline": _build_timeline(logs, task["status"], task["task_type"]),
        "logs": [_map_log_entry(log, idx) for idx, log in enumerate(logs, start=1)],
        "tuningFields": tuning_fields,
        "revisions": revisions,
    }


def build_workbench_snapshot(db_path: str | Path | None = None) -> Dict[str, Any]:
    raw_items = list_raw_datasets(db_path=db_path, limit=300)
    processed_items = list_preprocessed_datasets(db_path=db_path, limit=300)
    tasks = list_task_runs(db_path=db_path, limit=300)

    return {
        "modules": [
            {"label": "数据接入层", "status": "已可演示", "detail": "上传、历史回读、字段基础字典已形成原型。"},
            {"label": "预处理层", "status": "已可演示", "detail": "字段纠偏、缺失值、异常值、编码、缩放已打通。"},
            {"label": "统计分析层", "status": "已可演示", "detail": "EDA、统计检验、诊断与自动洞察已可用。"},
            {"label": "建模层", "status": "基础可用", "detail": "OLS、Logistic、随机森林、梯度提升已接入。"},
            {"label": "API 接口层", "status": "已接入首批接口", "detail": "任务总览、任务中心、运行监控、字段字典与建模运行已可通过 FastAPI 暴露。"},
        ],
        "datasets": [
            {"label": "原始数据记录", "value": len(raw_items)},
            {"label": "预处理结果版本", "value": len(processed_items)},
            {"label": "任务快照数", "value": len(tasks)},
        ],
        "nextActions": [
            "让上传数据集、预处理启动和建模启动全部走真实 FastAPI 接口。",
            "为日志、字段修正和重跑动作补充权限与审计链路。",
            "把 SQLite 原型逐步迁移到 PostgreSQL，并保留当前数据结构。",
        ],
        "apiContracts": [
            {"name": "/api/overview", "method": "GET", "status": "已接入"},
            {"name": "/api/datasets", "method": "GET", "status": "已接入"},
            {"name": "/api/datasets/upload", "method": "POST", "status": "已接入"},
            {"name": "/api/datasets/{dataset_id}/preview", "method": "GET", "status": "已接入"},
            {"name": "/api/preprocess/run", "method": "POST", "status": "已接入"},
            {"name": "/api/tasks/summary", "method": "GET", "status": "已接入"},
            {"name": "/api/tasks/{task_id}/monitoring", "method": "GET", "status": "已接入"},
            {"name": "/api/tasks/{task_id}/logs", "method": "GET", "status": "已接入"},
            {"name": "/api/datasets/{dataset_id}/dictionary", "method": "GET/PATCH", "status": "已接入"},
            {"name": "/api/modeling/run", "method": "POST", "status": "已接入"},
        ],
    }


def build_dataset_catalog(db_path: str | Path | None = None) -> list[Dict[str, Any]]:
    raw_items = list_raw_datasets(db_path=db_path, limit=500)
    processed_items = list_preprocessed_datasets(db_path=db_path, limit=500)
    latest_processed_map: dict[str, Dict[str, Any]] = {}

    for item in processed_items:
        latest_processed_map.setdefault(item["dataset_id"], item)

    return [
        {
            "datasetId": item["dataset_id"],
            "datasetName": item["dataset_name"],
            "sourceFileName": item.get("source_file_name"),
            "uploadTime": item["upload_time"],
            "rowCount": item["row_count"],
            "columnCount": item["column_count"],
            "latestPreprocessRunId": (latest_processed_map.get(item["dataset_id"]) or {}).get("preprocess_run_id"),
            "latestPreprocessTime": (latest_processed_map.get(item["dataset_id"]) or {}).get("preprocess_time"),
        }
        for item in raw_items
    ]


def build_dataset_preview(
    dataset_id: str,
    *,
    view: str = "raw",
    limit: int = 10,
    db_path: str | Path | None = None,
) -> Dict[str, Any]:
    dataset_entry = get_raw_dataset_entry(dataset_id, db_path=db_path)
    latest_preprocessed = _find_latest_preprocessed_entry(dataset_id, db_path=db_path)

    if view == "modeling" and latest_preprocessed is not None:
        df = load_preprocessed_dataset(latest_preprocessed["preprocess_run_id"], db_path=db_path)
        source_kind = "preprocessed_data"
        preprocess_run_id = latest_preprocessed["preprocess_run_id"]
    else:
        df = load_raw_dataset(dataset_id, db_path=db_path)
        source_kind = "raw_dataset"
        preprocess_run_id = None

    return {
        "datasetId": dataset_id,
        "datasetName": dataset_entry["dataset_name"],
        "sourceKind": source_kind,
        "preprocessRunId": preprocess_run_id,
        "rowCount": int(df.shape[0]),
        "columnCount": int(df.shape[1]),
        "columns": list(df.columns),
        "rows": _build_preview_rows(df, limit=limit),
    }


def build_dataset_dictionary(dataset_id: str, db_path: str | Path | None = None) -> list[Dict[str, Any]]:
    base_fields = _build_base_dictionary_fields(dataset_id, db_path=db_path)
    override_map = {
        item["field_name"]: item
        for item in list_dataset_dictionary_overrides(dataset_id, db_path=db_path)
    }

    merged_fields: list[Dict[str, Any]] = []
    for item in base_fields:
        override = override_map.get(item["fieldName"])
        if override:
            item = {
                **item,
                "dataType": override.get("data_type") or item["dataType"],
                "semanticType": override.get("semantic_type") or item["semanticType"],
                "unit": override.get("unit") or item["unit"],
                "softRange": override.get("soft_range") or item["softRange"],
                "note": override.get("note") or item["note"],
            }
        merged_fields.append(item)

    return merged_fields


def build_dataset_dictionary_revisions(dataset_id: str, db_path: str | Path | None = None) -> list[Dict[str, Any]]:
    rows = list_dataset_dictionary_revisions(dataset_id, db_path=db_path, limit=50)
    return [
        {
            "id": f"REV-{index}",
            "time": row["created_time"],
            "actor": row.get("actor") or "人工微调",
            "fieldName": row["field_name"],
            "summary": _build_revision_summary(row),
        }
        for index, row in enumerate(rows, start=1)
    ]


def save_dataset_dictionary_field(
    dataset_id: str,
    *,
    field_name: str,
    data_type: str | None = None,
    semantic_type: str | None = None,
    unit: str | None = None,
    soft_range: str | None = None,
    note: str | None = None,
    summary: str | None = None,
    actor: str | None = None,
    task_id: str | None = None,
    db_path: str | Path | None = None,
) -> Dict[str, Any]:
    override_entry = upsert_dataset_dictionary_override(
        dataset_id,
        field_name=field_name,
        data_type=data_type,
        semantic_type=semantic_type,
        unit=unit,
        soft_range=soft_range,
        note=note,
        db_path=db_path,
    )
    revision_entry = create_dataset_dictionary_revision(
        dataset_id,
        field_name=field_name,
        data_type=data_type,
        semantic_type=semantic_type,
        unit=unit,
        soft_range=soft_range,
        note=summary or note,
        actor=actor,
        db_path=db_path,
    )

    if task_id:
        append_task_step_log(
            task_id,
            step_name="人工微调",
            step_status="done",
            log_level="warning",
            message=f"字段“{field_name}”已被人工更新。",
            detail={"dataset_id": dataset_id, "revision": revision_entry},
            db_path=db_path,
        )

    return {
        "override": override_entry,
        "revision": revision_entry,
        "dictionary": build_dataset_dictionary(dataset_id, db_path=db_path),
        "revisions": build_dataset_dictionary_revisions(dataset_id, db_path=db_path),
    }


def _build_base_dictionary_fields(dataset_id: str, db_path: str | Path | None = None) -> list[Dict[str, Any]]:
    preprocessed_entry = _find_latest_preprocessed_entry(dataset_id, db_path=db_path)
    if preprocessed_entry is not None:
        resolved_types = preprocessed_entry["resolved_types_json"]
        field_metadata = preprocessed_entry["field_metadata_json"]
        fields: list[Dict[str, Any]] = []
        for column, base_type in resolved_types.items():
            metadata = field_metadata.get(column, {})
            fields.append(
                {
                    "id": f"{dataset_id}:{column}",
                    "fieldName": column,
                    "dataType": TYPE_LABELS.get(base_type, base_type),
                    "semanticType": _resolve_semantic_label(metadata, base_type),
                    "unit": _resolve_unit(metadata, base_type),
                    "softRange": _resolve_soft_range(metadata, base_type),
                    "note": metadata.get("description") or "",
                }
            )
        return fields

    raw_df = load_raw_dataset(dataset_id, db_path=db_path)
    inferred_types = infer_column_types(raw_df)
    return [
        {
            "id": f"{dataset_id}:{column}",
            "fieldName": column,
            "dataType": TYPE_LABELS.get(inferred_types.get(column, "categorical"), "分类"),
            "semanticType": _default_semantic_for_type(inferred_types.get(column, "categorical")),
            "unit": "年" if inferred_types.get(column) == "datetime" else "无",
            "softRange": "未设置",
            "note": "",
        }
        for column in raw_df.columns
    ]


def _find_latest_preprocessed_entry(dataset_id: str, db_path: str | Path | None = None) -> Dict[str, Any] | None:
    processed_items = list_preprocessed_datasets(db_path=db_path, limit=500)
    for item in processed_items:
        if item["dataset_id"] == dataset_id:
            return get_preprocessed_entry(item["preprocess_run_id"], db_path=db_path)
    return None


def _build_recent_task_trend(tasks: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    bucket = defaultdict(lambda: {"total": 0, "preprocess": 0, "modeling": 0})
    today = datetime.now().date()
    for offset in range(6, -1, -1):
        label = (today - timedelta(days=offset)).strftime("%m-%d")
        bucket[label]

    for task in tasks:
        task_date = _parse_time(task["created_time"]).strftime("%m-%d")
        if task_date not in bucket:
            continue
        bucket[task_date]["total"] += 1
        if task["task_type"] == "preprocess":
            bucket[task_date]["preprocess"] += 1
        if task["task_type"] == "modeling":
            bucket[task_date]["modeling"] += 1

    return [
        {"label": label, **values}
        for label, values in sorted(bucket.items())
    ]


def _build_model_ranking(tasks: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    scores: dict[str, list[float]] = defaultdict(list)
    for task in tasks:
        if task["task_type"] != "modeling" or task["status"] != "completed":
            continue
        payload = task.get("result_payload_json") or {}
        metrics = payload.get("metrics") or {}
        algorithm = payload.get("algorithm") or "未知模型"
        if "r2" in metrics:
            scores[algorithm].append(float(metrics["r2"]))
        elif "accuracy" in metrics:
            scores[algorithm].append(float(metrics["accuracy"]))

    if not scores:
        return []

    ranking = [
        {"label": name, "score": round(sum(values) / len(values), 4)}
        for name, values in scores.items()
    ]
    ranking.sort(key=lambda item: item["score"], reverse=True)
    return ranking[:10]


def _build_recent_alerts(
    tasks: list[Dict[str, Any]],
    raw_items: list[Dict[str, Any]],
    *,
    db_path: str | Path | None = None,
) -> list[Dict[str, Any]]:
    alerts: list[Dict[str, Any]] = []
    raw_name_map = {item["dataset_id"]: item["dataset_name"] for item in raw_items}

    for task in tasks[:30]:
        logs = list_task_step_logs(task["task_id"], db_path=db_path, limit=20)
        for log in logs:
            if log["log_level"] not in {"warning", "error"}:
                continue
            field_name = log["detail_json"].get("field_name")
            alerts.append(
                {
                    "title": log["message"],
                    "detail": raw_name_map.get(task.get("dataset_id"), task.get("source_name") or "未绑定数据集"),
                    "severity": "高" if log["log_level"] == "error" else "中",
                }
            )
            if len(alerts) >= 5:
                return alerts
        if task["status"] == "failed" and len(alerts) < 5:
            alerts.append(
                {
                    "title": f"{_build_task_name(task, raw_name_map)}执行失败",
                    "detail": task.get("error_message") or "请进入运行监控页查看失败细节。",
                    "severity": "高",
                }
            )
    return alerts[:5]


def _build_task_item(
    task: Dict[str, Any],
    raw_items: list[Dict[str, Any]],
    *,
    db_path: str | Path | None = None,
) -> Dict[str, Any]:
    raw_name_map = {item["dataset_id"]: item["dataset_name"] for item in raw_items}
    logs = list_task_step_logs(task["task_id"], db_path=db_path, limit=100)
    latest_log = logs[-1] if logs else None
    record_count = (
        (task.get("request_payload_json") or {}).get("row_count")
        or (task.get("result_payload_json") or {}).get("sample_size")
        or 0
    )
    version = (
        (task.get("result_payload_json") or {}).get("preprocess_run_id")
        or (find_preprocessed_entry_by_task_id(task["task_id"], db_path=db_path) or {}).get("preprocess_run_id")
        or "v1"
    )

    return {
        "id": task["task_id"],
        "name": _build_task_name(task, raw_name_map),
        "type": task["task_type"],
        "dataset": raw_name_map.get(task.get("dataset_id"), task.get("source_name") or "未绑定数据集"),
        "datasetId": task.get("dataset_id"),
        "status": task["status"],
        "owner": "系统代理",
        "startedAt": task["created_time"],
        "duration": _format_duration(task["created_time"], task["updated_time"], task["status"]),
        "stage": latest_log["step_name"] if latest_log else _status_label(task["status"]),
        "records": f"{record_count:,}" if isinstance(record_count, int | float) else str(record_count),
        "version": str(version),
        "alerts": sum(1 for log in logs if log["log_level"] in {"warning", "error"}),
    }


def _build_task_name(task: Dict[str, Any], raw_name_map: Dict[str, str]) -> str:
    base = TASK_TYPE_LABELS.get(task["task_type"], task["task_type"])
    suffix = raw_name_map.get(task.get("dataset_id"), task.get("source_name") or task["task_id"][:8])
    return f"{suffix} {base}"


def _build_timeline(logs: list[Dict[str, Any]], task_status: str, task_type: str) -> list[Dict[str, Any]]:
    if not logs:
        return _default_timeline(task_status, task_type)

    seen_steps: dict[str, Dict[str, Any]] = {}
    for log in logs:
        seen_steps[log["step_name"]] = {
            "name": log["step_name"],
            "status": log["step_status"],
            "duration": "已记录",
            "detail": log["message"],
        }
    return list(seen_steps.values())


def _default_timeline(task_status: str, task_type: str) -> list[Dict[str, Any]]:
    if task_type == "preprocess":
        steps = ["载入数据集", "自动识别字段类型", "字段约束校验", "特征转换", "结果落库"]
    else:
        steps = ["准备特征矩阵", "模型训练", "结果解释", "结果归档"]

    mapped_status = "done" if task_status == "completed" else "failed" if task_status == "failed" else "working"
    return [
        {"name": step, "status": mapped_status if index == 0 else "waiting", "duration": "--", "detail": "暂无步骤日志。"}
        for index, step in enumerate(steps)
    ]


def _map_log_entry(log: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "id": f"LOG-{index}",
        "time": _parse_time(log["created_time"]).strftime("%H:%M:%S"),
        "level": LOG_LEVEL_LABELS.get(log["log_level"], "信息"),
        "module": log["step_name"],
        "message": log["message"],
        "field": (log.get("detail_json") or {}).get("field_name"),
    }


def _synthesize_task_logs(task: Dict[str, Any]) -> list[Dict[str, Any]]:
    request_payload = task.get("request_payload_json") or {}
    result_payload = task.get("result_payload_json") or {}
    logs = [
        {
            "task_id": task["task_id"],
            "step_name": "任务初始化",
            "step_status": "done",
            "log_level": "info",
            "message": f"任务已创建，类型为 {task['task_type']}。",
            "detail_json": request_payload,
            "created_time": task["created_time"],
        }
    ]
    if result_payload:
        logs.append(
            {
                "task_id": task["task_id"],
                "step_name": "结果摘要",
                "step_status": "done" if task["status"] == "completed" else "failed",
                "log_level": "info" if task["status"] == "completed" else "error",
                "message": "任务已完成结果归档。" if task["status"] == "completed" else (task.get("error_message") or "任务执行失败。"),
                "detail_json": result_payload,
                "created_time": task["updated_time"],
            }
        )
    return logs


def _build_revision_summary(row: Dict[str, Any]) -> str:
    parts = []
    if row.get("data_type"):
        parts.append(f"类型改为 {row['data_type']}")
    if row.get("semantic_type"):
        parts.append(f"语义改为 {row['semantic_type']}")
    if row.get("unit"):
        parts.append(f"单位设为 {row['unit']}")
    if row.get("soft_range"):
        parts.append(f"范围设为 {row['soft_range']}")
    if row.get("note"):
        parts.append(row["note"])
    return "；".join(parts) if parts else "更新了字段元数据。"


def _resolve_semantic_label(metadata: Dict[str, Any], base_type: str) -> str:
    if metadata.get("custom_semantic_type"):
        return str(metadata["custom_semantic_type"])
    semantic_type = metadata.get("semantic_type")
    if semantic_type:
        return SEMANTIC_LABELS.get(str(semantic_type), str(semantic_type))
    return _default_semantic_for_type(base_type)


def _resolve_unit(metadata: Dict[str, Any], base_type: str) -> str:
    return (
        metadata.get("unit")
        or metadata.get("canonical_unit")
        or ("年" if base_type == "datetime" else "无")
    )


def _resolve_soft_range(metadata: Dict[str, Any], base_type: str) -> str:
    if base_type == "datetime":
        start = metadata.get("soft_start_time") or metadata.get("legal_start_time")
        end = metadata.get("soft_end_time") or metadata.get("legal_end_time")
        if start or end:
            return f"{start or '未设置'} ~ {end or '未设置'}"
        granularity = metadata.get("datetime_granularity")
        return str(granularity) if granularity else "未设置"

    soft_min = metadata.get("soft_min")
    soft_max = metadata.get("soft_max")
    if soft_min is not None or soft_max is not None:
        return f"{soft_min if soft_min is not None else '未设置'} ~ {soft_max if soft_max is not None else '未设置'}"
    raw_scale = metadata.get("raw_scale")
    return str(raw_scale) if raw_scale else "未设置"


def _default_semantic_for_type(base_type: str) -> str:
    if base_type == "numerical":
        return "普通字段"
    if base_type == "categorical":
        return "普通分类"
    if base_type == "datetime":
        return "时间索引"
    if base_type == "boolean":
        return "布尔标签"
    if base_type == "text":
        return "文本标签"
    return "普通字段"


def _build_preview_rows(df, *, limit: int) -> list[Dict[str, Any]]:
    preview_df = df.head(limit).copy()
    return json.loads(preview_df.to_json(orient="records", force_ascii=False, date_format="iso"))


def _format_duration(created_time: str, updated_time: str, status: str) -> str:
    start = _parse_time(created_time)
    end = _parse_time(updated_time) if status in {"completed", "failed"} else datetime.now()
    delta = max(end - start, timedelta(seconds=0))
    total_seconds = int(delta.total_seconds())
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _status_label(status: str) -> str:
    return {
        "completed": "已完成",
        "running": "运行中",
        "failed": "失败",
        "queued": "排队中",
    }.get(status, status)
