from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.datetime_utils import parse_datetime_series
from app.core.eda_stats import (
    compute_vif_table,
    generate_eda_insights,
    run_statistical_diagnostic,
    run_statistical_test,
)
from app.core.type_infer import infer_column_types
from app.schemas.config import ModelingConfig, PreprocessConfig, SplitConfig
from app.services.modeling_service import run_modeling_with_source
from app.services.preprocess_service import (
    persist_preprocessed_dataframe,
    persist_raw_dataframe,
    run_preprocess_with_source,
)
from app.storage.sqlite_store import get_task_run, load_preprocessed_dataset, load_raw_dataset
from app.storage.sqlite_store import find_preprocessed_entry_by_task_id, list_task_runs


def main() -> None:
    type_probe = pd.DataFrame(
        {
            "current_hdi": [0.62, 0.64, 0.66],
            "current_policy_intensity_total": [12.1, 13.5, 15.2],
            "signup_date": ["2024-01-01", "2024-01-05", "2024-01-10"],
        }
    )
    inferred = infer_column_types(type_probe)
    assert inferred["current_hdi"] == "numerical"
    assert inferred["current_policy_intensity_total"] == "numerical"
    assert inferred["signup_date"] == "datetime"

    year_probe = pd.DataFrame(
        {
            "year": [2011, 2015, 2020],
            "month_id": [202001, 202002, 202003],
        }
    )
    parsed_year = parse_datetime_series(year_probe["year"], "year")
    assert str(parsed_year.min().date()) == "2011-01-01"
    inferred_year = infer_column_types(year_probe)
    assert inferred_year["year"] == "datetime"
    assert inferred_year["month_id"] == "datetime"

    stats_probe = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B"],
            "score": [10, 11, 9, 20, 21, 22],
            "x": [1, 2, 3, 4, 5, 6],
            "y": [2, 4, 6, 8, 10, 12],
            "year": [2011, 2012, 2013, 2014, 2015, 2016],
        }
    )
    inferred_stats = infer_column_types(stats_probe)
    t_result = run_statistical_test(stats_probe, inferred_stats, "group", "score")
    assert t_result["status"] == "ok"
    assert t_result["method"] == "Welch t 检验"
    pearson_result = run_statistical_test(stats_probe, inferred_stats, "x", "y")
    assert pearson_result["status"] == "ok"
    assert pearson_result["significant"] is True
    diag_result = run_statistical_diagnostic(stats_probe, inferred_stats, "group", "score")
    assert diag_result["status"] == "ok"
    assert diag_result["recommended_test"] in {"独立样本 t 检验", "Welch t 检验", "Mann-Whitney U 检验"}
    insights = generate_eda_insights(
        stats_probe,
        {"inferred_types": inferred_stats, "quality_report": {"column_profiles": []}},
    )
    assert isinstance(insights, list)

    vif_probe = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6],
            "x2": [2, 4, 6, 8, 10, 12],
            "x3": [1, 1, 2, 3, 5, 8],
        }
    )
    vif_types = infer_column_types(vif_probe)
    vif_table = compute_vif_table(vif_probe, vif_types)
    assert not vif_table.empty
    assert "VIF" in vif_table.columns

    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 3],
            "age": [23, None, 35, 35],
            "gender": ["M", "F", None, None],
            "signup_date": ["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-10"],
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_save = persist_raw_dataframe(
            df,
            dataset_name="smoke_dataset",
            source_file_name="smoke.csv",
        )
        config = PreprocessConfig(
            drop_columns=["user_id"],
            missing_numeric="median",
            missing_categorical="mode",
            outlier_strategy="clip_iqr",
            encoding="onehot",
            scaling="standard",
            save_artifacts=True,
            split=SplitConfig(test_size=0.25, random_state=42),
        )

        result = run_preprocess_with_source(
            df,
            config,
            source_name="smoke.csv",
            artifact_root=tmp_dir,
            dataset_id=raw_save["dataset_id"],
        )
        print("analysis_columns", len(result["analysis"]["inferred_types"]))
        print("cleaned_shape", result["cleaned_df"].shape)
        print("transformed_shape", result["transformed_df"].shape)
        print("split_keys", sorted(result["splits"].keys()))
        print("artifact_dir_name", Path(result["artifacts"]["run_dir"]).name)

        task_info = result["task"]
        assert task_info["status"] == "completed"
        stored_task = get_task_run(task_info["task_id"])
        assert stored_task["status"] == "completed"
        assert stored_task["result_payload_json"]["status"] == "completed"
        assert stored_task["result_payload_json"]["preprocess_run_id"] is None

        preprocess_save = persist_preprocessed_dataframe(
            result["transformed_df"],
            dataset_id=raw_save["dataset_id"],
            resolved_types=result["resolved_types"],
            field_metadata={},
            preprocess_config=config.model_dump(),
            validation_findings=result["validation_findings"],
        )
        loaded_raw_df = load_raw_dataset(raw_save["dataset_id"])
        loaded_preprocessed_df = load_preprocessed_dataset(preprocess_save["preprocess_run_id"])

        assert Path(result["artifacts"]["files"]["quality_report"]).exists()
        assert Path(result["artifacts"]["files"]["preprocess_config"]).exists()
        assert Path(result["artifacts"]["files"]["transformed_data"]).exists()
        assert loaded_raw_df.shape == df.shape
        assert loaded_preprocessed_df.shape == result["transformed_df"].shape

        year_result = run_preprocess_with_source(
            pd.DataFrame({"year": [2011, 2012, 2020], "value": [10, 20, 30]}),
            PreprocessConfig(save_artifacts=False),
            source_name="year_smoke.csv",
            artifact_root=tmp_dir,
        )
        assert str(year_result["cleaned_df"]["year"].min().date()) == "2011-01-01"
        assert year_result["transformed_df"]["year_year"].tolist() == [2011, 2012, 2020]
        assert year_result["task"]["status"] == "completed"

        integrated_result = run_preprocess_with_source(
            df,
            config,
            source_name="smoke.csv",
            artifact_root=tmp_dir,
            dataset_id=raw_save["dataset_id"],
            persist_output=True,
            field_metadata={"age": {"unit": "岁"}},
        )
        assert integrated_result["preprocess_storage"] is not None
        assert integrated_result["preprocess_storage"]["task_id"] == integrated_result["task"]["task_id"]

        integrated_task = get_task_run(integrated_result["task"]["task_id"])
        assert (
            integrated_task["result_payload_json"]["preprocess_run_id"]
            == integrated_result["preprocess_storage"]["preprocess_run_id"]
        )
        linked_preprocessed_entry = find_preprocessed_entry_by_task_id(integrated_result["task"]["task_id"])
        assert linked_preprocessed_entry is not None
        assert linked_preprocessed_entry["preprocess_run_id"] == integrated_result["preprocess_storage"]["preprocess_run_id"]

        completed_tasks = list_task_runs(
            task_type="preprocess",
            dataset_id=raw_save["dataset_id"],
            status="completed",
            limit=20,
        )
        assert completed_tasks
        assert any(task["task_id"] == integrated_result["task"]["task_id"] for task in completed_tasks)

        modeling_df = pd.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5, 6, 7, 8],
                "x2": [2, 1, 4, 3, 6, 5, 8, 7],
                "group": ["A", "A", "A", "B", "B", "B", "B", "A"],
                "target": [3, 5, 7, 9, 11, 13, 15, 17],
                "label": [0, 0, 0, 1, 1, 1, 1, 0],
            }
        )

        ols_result = run_modeling_with_source(
            modeling_df,
            ModelingConfig(
                model_family="statistical",
                algorithm="ols",
                problem_type="regression",
                target_column="target",
                feature_columns=["x1", "x2"],
                save_artifacts=False,
            ),
            source_name="modeling_smoke.csv",
            dataset_id=raw_save["dataset_id"],
        )
        assert ols_result["task"]["status"] == "completed"
        assert "r2" in ols_result["result"]["metrics"]
        assert ols_result["prediction_df"].shape[0] > 0

        rf_result = run_modeling_with_source(
            modeling_df,
            ModelingConfig(
                model_family="machine_learning",
                algorithm="random_forest",
                problem_type="regression",
                target_column="target",
                feature_columns=["x1", "x2", "group"],
                save_artifacts=False,
                n_estimators=50,
                max_depth=4,
            ),
            source_name="modeling_smoke.csv",
            dataset_id=raw_save["dataset_id"],
        )
        assert rf_result["task"]["status"] == "completed"
        assert "mae" in rf_result["result"]["metrics"]
        assert rf_result["prediction_df"].shape[0] > 0

        modeling_tasks = list_task_runs(
            task_type="modeling",
            dataset_id=raw_save["dataset_id"],
            status="completed",
            limit=20,
        )
        assert modeling_tasks
        assert any(task["task_id"] == ols_result["task"]["task_id"] for task in modeling_tasks)
        assert any(task["task_id"] == rf_result["task"]["task_id"] for task in modeling_tasks)


if __name__ == "__main__":
    main()
