from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.fastapi_app import create_app


CSV_PAYLOAD = """city_group,emission_ratio,year,energy_intensity,target
长三角,12.5,2017,0.62,82
珠三角,14.2,2018,0.64,84
成渝,10.8,2019,0.59,79
长三角,13.1,2020,0.58,86
珠三角,15.0,2021,0.61,88
成渝,11.3,2022,0.57,81
长三角,12.9,2023,0.56,89
珠三角,14.8,2024,0.60,87
"""


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "api_smoke.db"
        client = TestClient(create_app(db_path=db_path))

        health_response = client.get("/api/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "ok"

        upload_response = client.post(
            "/api/datasets/upload",
            data={"dataset_name": "api_smoke_dataset"},
            files={"file": ("api_smoke.csv", io.BytesIO(CSV_PAYLOAD.encode("utf-8-sig")), "text/csv")},
        )
        assert upload_response.status_code == 200
        upload_payload = upload_response.json()
        dataset_id = upload_payload["datasetId"]
        assert upload_payload["rowCount"] == 8
        assert upload_payload["columnCount"] == 5

        datasets_response = client.get("/api/datasets")
        assert datasets_response.status_code == 200
        datasets_payload = datasets_response.json()
        assert datasets_payload["items"]
        assert any(item["datasetId"] == dataset_id for item in datasets_payload["items"])

        raw_preview_response = client.get(f"/api/datasets/{dataset_id}/preview?view=raw&limit=5")
        assert raw_preview_response.status_code == 200
        raw_preview = raw_preview_response.json()
        assert raw_preview["sourceKind"] == "raw_dataset"
        assert len(raw_preview["rows"]) == 5
        assert "target" in raw_preview["columns"]

        preprocess_response = client.post(
            "/api/preprocess/run",
            json={
                "datasetId": dataset_id,
                "dropColumns": [],
                "missingNumeric": "median",
                "missingCategorical": "mode",
                "outlierStrategy": "none",
                "encoding": "onehot",
                "scaling": "none",
                "removeDuplicates": True,
                "dropAllNullColumns": True,
                "dropSingleValueColumns": True,
                "saveArtifacts": False,
                "enableSplit": False,
                "splitTestSize": 0.2,
                "fieldMetadata": {
                    "city_group": {"semantic_type": "geographic_region", "description": "城市群分组字段"},
                    "emission_ratio": {"semantic_type": "proportion", "unit": "%", "soft_min": 0, "soft_max": 100},
                    "year": {"datetime_granularity": "year", "unit": "年", "soft_start_time": "2017-01-01", "soft_end_time": "2024-12-31"},
                },
            },
        )
        assert preprocess_response.status_code == 200
        preprocess_payload = preprocess_response.json()
        assert preprocess_payload["taskStatus"] == "completed"
        assert preprocess_payload["preprocessRunId"]
        assert preprocess_payload["preview"]["sourceKind"] == "preprocessed_data"

        task_id = preprocess_payload["taskId"]
        monitoring_response = client.get(f"/api/tasks/{task_id}/monitoring")
        assert monitoring_response.status_code == 200
        monitoring_payload = monitoring_response.json()
        assert monitoring_payload["task"]["id"] == task_id
        assert monitoring_payload["logs"]

        dictionary_response = client.get(f"/api/datasets/{dataset_id}/dictionary")
        assert dictionary_response.status_code == 200
        dictionary_payload = dictionary_response.json()
        assert dictionary_payload["fields"]

        patch_response = client.patch(
            f"/api/datasets/{dataset_id}/dictionary",
            json={
                "fieldName": "city_group",
                "dataType": "分类",
                "semanticType": "地理区域",
                "unit": "不适用",
                "softRange": "不适用",
                "note": "保留城市群层级标记",
                "summary": "确认 city_group 属于地理区域语义，并禁用量纲配置。",
                "actor": "API 烟测",
                "taskId": task_id,
            },
        )
        assert patch_response.status_code == 200
        patch_payload = patch_response.json()
        assert patch_payload["revisions"][0]["fieldName"] == "city_group"

        modeling_preview_response = client.get(f"/api/datasets/{dataset_id}/preview?view=modeling&limit=5")
        assert modeling_preview_response.status_code == 200
        modeling_preview = modeling_preview_response.json()
        assert modeling_preview["sourceKind"] == "preprocessed_data"
        assert "target" in modeling_preview["columns"]

        feature_columns = [column for column in modeling_preview["columns"] if column != "target"]
        modeling_response = client.post(
            "/api/modeling/run",
            json={
                "datasetId": dataset_id,
                "preprocessRunId": preprocess_payload["preprocessRunId"],
                "config": {
                    "model_family": "machine_learning",
                    "algorithm": "random_forest",
                    "problem_type": "regression",
                    "target_column": "target",
                    "feature_columns": feature_columns,
                    "test_size": 0.2,
                    "save_artifacts": False,
                    "n_estimators": 40,
                    "max_depth": 4,
                },
            },
        )
        assert modeling_response.status_code == 200
        modeling_payload = modeling_response.json()
        assert modeling_payload["task"]["status"] == "completed"
        assert modeling_payload["result"]["metrics"]

        logs_response = client.get(f"/api/tasks/{modeling_payload['task']['task_id']}/logs")
        assert logs_response.status_code == 200
        assert logs_response.json()["logs"]

        overview_response = client.get("/api/overview")
        assert overview_response.status_code == 200
        overview_payload = overview_response.json()
        assert overview_payload["heroStats"]

        print("api_health", health_response.json()["status"])
        print("dataset_count", len(datasets_payload["items"]))
        print("preprocess_task", preprocess_payload["taskId"])
        print("model_task", modeling_payload["task"]["task_id"])
        print("model_metrics", sorted(modeling_payload["result"]["metrics"].keys()))


if __name__ == "__main__":
    main()
